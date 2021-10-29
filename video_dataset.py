import numpy as np
import glob
import utils
import torch
import random
import copy


class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_features = self.dataset_name + '-I3D-JOINTFeatures.npy'
        print(self.path_to_features)
        self.path_to_annotations = self.dataset_name + '-Annotations/'
        self.features = np.load(self.path_to_features, encoding='bytes',allow_pickle=True)
        self.segments = np.load(self.path_to_annotations + 'segments.npy',allow_pickle=True)
        self.gtlabels = np.load(self.path_to_annotations + 'labels.npy',allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy',allow_pickle=True)
        self.activity_net = args.activity_net
        # self.max_labels = args.max_lab
        if not self.activity_net:
            self.labels101 = np.load('Thumos14-Annotations/labels.npy',allow_pickle=True)
            self.classlist101 = np.load('Thumos14-Annotations/classlist.npy',allow_pickle=True)
            self.classlist20 = np.load('Thumos14reduced-Annotations/classlist.npy',allow_pickle=True)
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy',allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + 'subset.npy',allow_pickle=True)
        self.duration = np.load(self.path_to_annotations + 'duration.npy',allow_pickle=True)
        self.videoname = np.load(self.path_to_annotations + 'videoname.npy',allow_pickle=True)
        self.lst_valid = None
        if self.activity_net:
            lst_valid = []
            for i in range(self.features.shape[0]):
                feat = self.features[i]
                mxlen = np.sum(np.max(np.abs(feat), axis=1) > 0, axis=0)
                if mxlen > 5:
                    lst_valid.append(i)
            self.lst_valid = lst_valid
            if len(lst_valid) != self.features.shape[0]:
                self.features = self.features[lst_valid]
                self.subset = self.subset[lst_valid]
                self.videoname = self.videoname[lst_valid]
                self.duration = self.duration[lst_valid]
                self.gtlabels = self.gtlabels[lst_valid]
                self.labels = self.labels[lst_valid]
                self.segments = self.segments[lst_valid]

        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs,self.classlist) for labs in self.labels]
        self.train_test_idx()
        
        self.classwise_feature_mapping()
        self.labels101to20 = np.array(self.classes101to20()) if args.num_class == 101 else None




    def train_test_idx(self):
        if not self.activity_net:
            test_str = 'validation'  # Thumos14
        else:
            test_str = 'training'    # ActivityNet

        for i, s in enumerate(self.subset):
            try: 
                si = s.decode('utf-8') 
            except: 
                si = s
            if si == test_str:   
                self.trainidx.append(i)
            else:
                self.testidx.append(i)



    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    try: 
                        cat = category.decode('utf-8') 
                    except: 
                        cat = category
                    if label == cat:
                        idx.append(i); break;
            self.classwiseidx.append(idx)


    def load_data(self, n_similar=0, is_training=True, multi_label=False, validation=False):
        if is_training==True:
            features = []
            labels = []
            idx = []
            t_max = self.t_max if not multi_label else 2*self.t_max
            
            # random sampling
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            feat = [self.features[i] for i in idx]
            lab = np.array([self.labels_multihot[i] for i in idx])
            max_seq_len = max([len(self.features[i]) for i in idx])            
            feat = np.array([utils.process_feat(feat[i],  min(t_max, max_seq_len)) for i in range(self.batch_size)])
            feat = np.array([feat[i] for i in range(self.batch_size)])

            return feat, lab

        else:
            idx = self.testidx[self.currenttestidx]
            labs = self.labels_multihot[idx]
            feat = self.features[idx]
                       
            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array([feat]), np.array(labs), done


    def classes101to20(self):

        classlist20 = np.array([c.decode('utf-8') for c in self.classlist20])
        classlist101 = np.array([c.decode('utf-8') for c in self.classlist101])
        labelsidx = []
        for categoryname in classlist20:
            labelsidx.append([i for i in range(len(classlist101)) if categoryname==classlist101[i]][0])
        
        return labelsidx


    
