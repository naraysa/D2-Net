import numpy as np


def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]


def filter_segments(segment_predict, videonames, ambilist):
   num_frames = 16 
   ind = np.zeros(np.shape(segment_predict)[0])
   for i in range(np.shape(segment_predict)[0]):
      vn = videonames[int(segment_predict[i,0])]
      for a in ambilist:
         if a[0]==vn:
            gt = range(int(round(float(a[2])*25/num_frames)), int(round(float(a[3])*25/num_frames)))
            pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
            IoU = float(len(set(gt).intersection(set(pd))))/float(len(set(gt).union(set(pd))))
            if IoU > 0:
               ind[i] = 1
   s = [segment_predict[i,:] for i in range(np.shape(segment_predict)[0]) if ind[i]==0]
   return np.array(s)

# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def nms_apply(proposals, iou_th=0.3):
   proposals_new = []
   proposals_l = [i for i in proposals]
   score_l = np.array([i[3] for i in proposals_l])
   idx = np.argsort(score_l)
   tmp = [proposals_l[i] for i in idx]
   done = False or (len(tmp)==0)
   while done != True:
      pop_t = tmp.pop()
      proposals_new.append(pop_t)
      p1 = range(pop_t[1],pop_t[2])            
      rm = []
      for i in range(len(tmp)):
         p2 = range(tmp[i][1],tmp[i][2])
         iou = float(len(set(p1).intersection(set(p2))))/float(len(set(p1).union(set(p2))))
         if iou >= iou_th:
            rm.append(1)
         else:
            rm.append(0)
      cur_id = len(rm)-1
      while len(rm) > 0:
         if rm.pop() == 1:
            del tmp[cur_id]
         cur_id -= 1
      if len(tmp) == 0:
         done = True

   return proposals_new


def getDetections(predictions, annotation_path, activity_net, valid_id):

   # gtsegments - temporal segments
   # gtlabels - labels for temporal segments
   # subset - test / validation string indicator for video
   gtsegments = np.load(annotation_path + '/segments.npy',allow_pickle=True)
   gtlabels = np.load(annotation_path + '/labels.npy',allow_pickle=True)
   videoname = np.load(annotation_path + '/videoname.npy',allow_pickle=True); 
   subset = np.load(annotation_path + '/subset.npy',allow_pickle=True); 
   classlist = np.load(annotation_path + '/classlist.npy',allow_pickle=True); 
   duration = np.load(annotation_path + '/duration.npy',allow_pickle=True)
   try: 
      classlist = np.array([c.decode('utf-8') for c in classlist]) 
      videoname = np.array([v.decode('utf-8') for v in videoname])
      subset = np.array([s.decode('utf-8') for s in subset])
   except: 
      classlist = np.array(classlist)
      videoname = np.array(videoname)
      subset = np.array(subset)
   

   if not activity_net:
      gtseg = np.load('Thumos14reduced-Annotations/test_gt_segments.npy',allow_pickle=True) 
      ambilist = annotation_path + '/Ambiguous_test.txt'
      ambilist = list(open(ambilist,'r'))
      ambilist = [a.strip('\n').split(' ') for a in ambilist]
   else:
      gtsegments = gtsegments[valid_id]
      gtlabels = gtlabels[valid_id]
      videoname = videoname[valid_id]
      subset = subset[valid_id]
      duration = duration[valid_id]

   
   # Keep only the test subset annotations
   gts, gtl, vn, dn = [], [], [], []
   test_str = 'test' if not activity_net  else 'validation'
   for i, s in enumerate(subset):
      if subset[i]==test_str:
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         if not activity_net:
            dn.append(duration[i,0])
         else:
            dn.append(duration[i])
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   duration = dn

   # keep ground truth and predictions for instances with temporal annotations
   gts, gtl, vn, pred, dn = [], [], [], [], []
   for i, s in enumerate(gtsegments):
      if len(s) > 0:
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         pred.append(predictions[i])
         dn.append(duration[i])
         
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   predictions = pred
   duration = dn
   
   print('Found', str(len(predictions)), 'videos')
   assert len(predictions) == len(gtlabels), 'Unequal predictions and GT'

   if not activity_net:
      vid2ind = dict()
      for k in range(212):
         vid2ind[videoname[k][11:]] = k
      gtseg2 = []
      for k in range(20):
         gtk = gtseg[k]
         gts = [[vid2ind[gtk[i][0]], gtk[i][1], gtk[i][2]] for i in range(len(gtk))]
         gtseg2.append(gts)
      gtseg = gtseg2

   # which categories have temporal labels ?
   templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

   # the number index for those categories.
   templabelidx = []
   for t in templabelcategories:
      templabelidx.append(str2ind(t,classlist))
   if len(predictions[0][0]) == 20:
      templabelidx = [i for i in range(20)]
             
   
   predictions_mod = []
   c_score = []
   ind_all = []
   for i in range(len(predictions)):
      pr = predictions[i]
      prp = - pr; [prp[:,i].sort() for i in range(np.shape(prp)[1])]; prp=-prp
      end_id = max(1, int(np.shape(prp)[0]/8)) if not activity_net else max(1,int(np.shape(prp)[0]/4))
      c_s = np.mean(prp[:end_id,:],axis=0)
      if not activity_net:
         c_s_th = max(np.max(c_s)/2,c_s[np.argsort(c_s)[-3]])
         ind = (c_s > np.max(c_s)/2)* (c_s > 0)
      else:
         act_th = 1 if np.sum(c_s > 1) else 0.5
         c_s_th = max(act_th,c_s[np.argsort(c_s)[-3]]) 
         ind = c_s > c_s_th 
      gtind = ind
      if len(gtlabels[i]) > 0:
         gtind = np.sum([classlist == g for g in gtlabels[i]],axis=0) > 0
      c_score.append(c_s)
      ind_all.append(ind)
      predictions_mod.append(pr*ind)
   predictions = predictions_mod



   detections_class = [] 
   gt_class = []
   mx_scores = [-100]*len(predictions)
   
   gtseg_c = -1
   for c in templabelidx:
      gtseg_c += 1
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         if ind_all[i][c] == 0: 
            continue
         tmp = predictions[i][:,c]
         segment_predict_i = []
         thresh_list = [i for i in range(1,20)]

         for threshold_wt in thresh_list:
            threshold = threshold_wt/40
            vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
            # start and end of proposals where segments are greater than the average threshold for the class
            s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
            e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
            for j in range(len(s)):
               aggr_score = 0
               # append proposal if length is at least 2 segments 
               if e[j]-s[j]>=2:     
                  # Outer-inner score
                  lt = max(1,int((e[j]-s[j])/4))
                  outer_score = None
                  if max(0,s[j]-lt) < s[j]:
                     outer_score = np.mean(tmp[max(0,s[j]-lt):s[j]])
                  if min(e[j]+lt,len(tmp)) > e[j]:
                     if outer_score is None:
                        outer_score = np.mean(tmp[e[j]:min(e[j]+lt,len(tmp))])
                     else:
                        outer_score = (outer_score + np.mean(tmp[e[j]:min(e[j]+lt,len(tmp))]))/2
                  inner_score = np.mean(tmp[s[j]:e[j]])
                  appendflag = False
                  if outer_score is not None:
                     aggr_score = inner_score - outer_score 
                     if  (outer_score - inner_score + 1)/2 <= 0.7:
                        appendflag = True
                  else:
                     aggr_score = inner_score 
                     appendflag = True
                  
                  if appendflag:
                     segment_predict_i.append([i, s[j], e[j], aggr_score])

         
         segment_predict_i = nms_apply(segment_predict_i)
         if len(segment_predict_i) > 0:
            max_sc = segment_predict_i[0][-1]
            if max_sc > mx_scores[i]:
               mx_scores[i] = max_sc
            segment_predict_i = [i for i in segment_predict_i if i[-1] >= 0.1*max_sc]
            
         cls_score_i = c_score[i][c]

         segment_predict += segment_predict_i

      segment_predict = np.array(segment_predict)
      if not activity_net:
         segment_predict = filter_segments(segment_predict, videoname, ambilist)
   
      # Create gt list    
      if activity_net:   
         segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
      else:
         segment_gt = gtseg[gtseg_c]

      detections_class.append(segment_predict) 
      gt_class.append(segment_gt)

   return detections_class, gt_class, templabelidx, duration
  


def getLocMAP(detections, gtsegments, temporal_labels, duration, iou, activity_net):

   ap = []
   div_f = 16
   num_c = 0
   fg_overlap, bg_overlap = 0, 0
   for c in temporal_labels:
      segment_predict = detections[num_c]
      if len(segment_predict) == 0:
         ap.append(0)
         continue
      # Sort the list of predictions for class c based on score
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]
      segment_gt = list(np.copy(gtsegments[num_c]))
      gtpos = len(segment_gt)
      num_c += 1
      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         flag = 0.
         best_iou = 0
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               if not activity_net:
                  vid_i = int(segment_gt[j][0])
                  gt = range(int(round(segment_gt[j][1]*duration[vid_i]*25/div_f)), int(round(segment_gt[j][2]*duration[vid_i]*25/div_f)))
               else:
                  gt = range(int(round(segment_gt[j][1]*25/div_f)), int(round(segment_gt[j][2]*25/div_f)))
               p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               # remove gt segment if IoU is greater than threshold 
               if IoU >= iou:
                  flag = 1.
                  if IoU > best_iou:
                     best_iou = IoU
                     best_j = j
                  del segment_gt[j]
                  break

         tp.append(flag)
         fp.append(1.-flag)
      
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if sum(tp)==0:
         prc = 0.
      else:
         cur_prec = tp_c / (fp_c+tp_c)
         cur_rec = 1. * tp_c / gtpos
         prc = _ap_from_pr(cur_prec, cur_rec)
         
      ap.append(prc)
      
   return 100*np.mean(ap)
  

def getDetectionMAP(predictions, annotation_path, activity_net=False, valid_id=None):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
   if activity_net:
      iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
   dmap_list = []
   detections, gtsegments, temporal_labels, duration = getDetections(predictions, annotation_path, activity_net, valid_id)
   

   for iou in iou_list:
      print('Testing for IoU %f' %iou)
      map_iou = getLocMAP(detections, gtsegments, temporal_labels, duration, iou, activity_net)
      dmap_list.append(map_iou)
      print('mAP %.4f' %(dmap_list[-1]))
    
   return dmap_list, iou_list 



