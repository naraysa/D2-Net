import numpy as np

def str2ind(categoryname,classlist):
   try:
     return [i for i in range(len(classlist)) if categoryname==classlist[i].decode('utf-8')][0]
   except:
     return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)

def interpolate_feat(feat):
   mid = np.expand_dims((feat[:-1] + feat[1:])/2,axis=1)
   feat1 = np.concatenate([np.expand_dims(feat[:-1],axis=1), mid], axis=1).reshape(len(mid)*2,len(feat[0]))
   return np.concatenate([feat1, feat[[-1]]], axis=0)

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)

def write_to_file(dname, dmap, itr):
    fid = open('logs/' + dname + '-results.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' %item
    fid.write(string_to_write + '\n')
    fid.close()


def write_summary(dname, summary):
    fid = open('logs/' + dname + '-results.log', 'a+')
    string_to_write = summary
    fid.write(string_to_write + '\n')
    fid.close()
