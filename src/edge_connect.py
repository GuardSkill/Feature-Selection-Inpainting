import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from libs.logger import Logger


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        val_sample = int(float((self.config.EVAL_INTERVAL)))
        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True, sample_interval=val_sample)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        # self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        log_path = os.path.join(config.PATH, 'logs_' + model_name)
        create_dir(log_path)
        self.logger = Logger(log_path)

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        # max_iteration = int(float((self.config.MAX_ITERS)))
        step_per_epoch = int(float((self.config.MAX_STEPS)))
        max_epoches = int(float((self.config.MAX_EPOCHES)))
        total = int(len(self.train_dataset))

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        print('\n\nThe number of Training data is %d' % total)

        while (keep_training):

            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(step_per_epoch, width=30, stateful_metrics=['iter'])

            for items in train_loader:
                self.edge_model.train()  # set the model to train mode
                self.inpaint_model.train()

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs['precision'] = precision.item()
                    logs['recall'] = recall.item()

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    if self.edge_model.iteration > step_per_epoch:
                        self.edge_model.iteration = 0
                    iteration = self.edge_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration


                if epoch > max_epoches:
                    keep_training = False
                    break
                # determine wether print the detail in training process
                # progbar.add(len(images)


                # log model at checkpoints

                if iteration == 0:
                    epoch += 1
                    print('\n\nTraining epoch: %d' % epoch)
                    progbar = Progbar(step_per_epoch, width=30, stateful_metrics=['epoch', 'iter'])
                    self.sample()
                    self.eval(epoch)
                    self.save()
                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value, epoch)
                else:
                    logs['step'] = iteration
                    progbar.add(1,
                                values=logs.items() if self.config.VERBOSE else [x for x in logs.items() if
                                                                                 not x[0].startswith('l_')])

        print('\nEnd training....')

    def eval(self,epoch):
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=False,
            num_workers=4
        )
        model = self.config.MODEL
        total = int(len(self.val_dataset))

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=30, stateful_metrics=['it'])
        iteration = 0

        for items in self.val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, _ = self.edge_model.process(images_gray, edges, masks)
                logs={}
                logs['l_val_d1']=dis_loss.item()
                logs['l_vak_g1']=gen_loss.item()
                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs['val_precision'] = precision.item()
                logs['val_recall'] = recall.item()
                for tag, value in logs.items():
                    self.logger.scalar_summary(tag, value, epoch)


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs

            logs["it"] = iteration
            progbar.add(len(images), values=logs.items())

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # edge-inpaint model
            elif model == 3:
                outputs = self.edge_model(images_gray, edges, masks).detach()
                outputs_merged = (outputs * masks) + (edges * (1 - masks)) / 2
                outputs_merged = self.postprocess(outputs_merged)[0]
                path = os.path.join(self.results_path + "/edge_inpainted", name)
                print(index, name)
                imsave(outputs_merged, path)

                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * masks) + (1 - masks)
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * (1-masks)) + (edges * masks)

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
