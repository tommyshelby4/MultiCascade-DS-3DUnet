from contextlib import redirect_stdout

def add_obj(objlIN, label, centroid, tomo_idx=None, cluster_size=None):
    obj = {
        'tomo_idx': tomo_idx,
        'label'   : label   ,
        'x'       :centroid[2] ,
        'y'       :centroid[1] ,
        'z'       :centroid[0] ,
        'cluster_size':cluster_size
    }
    # return objlIN.append(obj)
    objlIN.append(obj)
    return objlIN

def write_txt(objlIN, filename, classID2classname:dict):
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            for idx in range(len(objlIN)):
                lbl = objlIN[idx]['label']
                if(int(lbl) < 14):
                    lbl = classID2classname[str(int(lbl))]
                else:
                    lbl = classID2classname[str(lbl)]
                x = objlIN[idx]['x']
                y = objlIN[idx]['y']
                z = objlIN[idx]['z']
                csize = objlIN[idx]['cluster_size']
                if csize==None:
                    print(lbl.lower() + ' ' + str(z) + ' ' + str(y) + ' ' + str(x))
                else:
                    print(lbl.lower() + ' ' + str(z) + ' ' + str(y) + ' ' + str(x) + ' ' + str(csize))
    f.close()
