import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        
        self.model_name = 'model'
        self.log_dir = os.path.join(os.path.expanduser("~"), "tmp")
        
        self.log_path = os.path.join(self.log_dir, self.model_name)

        self.height = 192
        self.width = 640

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.scales = range(4)
        self.num_scales = 4
        
        self.frame_ids = [0, -1, 1]
        self.num_input_frames = len(self.opt.frame_ids)
        
        self.num_pose_frames = 2
        
        self.models["encoder"] = networks.ResnetEncoder(self.num_layers, pretrained=True)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
      
        self.models["pose"] = networks.PoseCNN(2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.learning_rate = 1e-4
        self.scheduler_step_size = 15
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.scheduler_step_size, 0.1)

        print("Training model named:\n  ", self.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_dir)
        print("Training is using:\n  ", self.device)

        self.dataset = datasets.KITTIRAWDataset

        fpath = os.path.join(os.path.dirname(__file__), "splits", "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png"

        num_train_samples = len(train_filenames)

        #TODO adjust?
        self.batch_size = 12
        self.num_epochs = 20
        
        self.min_depth = 0.1
        self.max_depth = 100
        
        self.num_total_steps = num_train_samples // self.batch_size * self.num_epochs


        self.data_path = '/content/drive/My Drive/ML/Projekat/monocular-depth-estimation/data/photos'

        train_dataset = self.dataset(
            self.data_path, train_filenames, self.height, self.width,
            self.frame_ids, 4, is_train=True, img_ext=img_ext)
        
        self.num_workers = 12 
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.data_path, val_filenames, self.height, self.width,
            self.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.batch_size, True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))


        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                # mean-normalized inverse depth from [62] to discourage shrinking of the estimated depth

                axisangle = outputs[("axisangle", 0, frame_id)]
                translation = outputs[("translation", 0, frame_id)]

                inv_depth = 1 / depth
                mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                T = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
    
#         photometric error function pe
    
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)


# We therefore set μ to only include the loss of pixels where the reprojection error
# of the warped image I t 0 →t is lower than that of the original, unwarped source image I t 0 , i.e
            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

                # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.batch_size)):  # write a maxmimum of four images
            for s in self.scales:
                for frame_id in self.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

#     def save_opts(self):
#         """Save options to disk so we know what we ran this experiment with
#         """
#         models_dir = os.path.join(self.log_path, "models")
#         if not os.path.exists(models_dir):
#             os.makedirs(models_dir)
#         to_save = self.opt.__dict__.copy()

#         with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
#             json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.height
                to_save['width'] = self.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

#     def load_model(self):
#         """Load model(s) from disk
#         """
#         self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

#         assert os.path.isdir(self.opt.load_weights_folder), \
#             "Cannot find folder {}".format(self.opt.load_weights_folder)
#         print("loading model from folder {}".format(self.opt.load_weights_folder))

#         for n in self.opt.models_to_load:
#             print("Loading {} weights...".format(n))
#             path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
#             model_dict = self.models[n].state_dict()
#             pretrained_dict = torch.load(path)
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             model_dict.update(pretrained_dict)
#             self.models[n].load_state_dict(model_dict)

#         # loading adam state
#         optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
#         if os.path.isfile(optimizer_load_path):
#             print("Loading Adam weights")
#             optimizer_dict = torch.load(optimizer_load_path)
#             self.model_optimizer.load_state_dict(optimizer_dict)
#         else:
#             print("Cannot find Adam weights so Adam is randomly initialized")