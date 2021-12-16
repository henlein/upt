import torch


if __name__ == "__main__":

    model1 = torch.load("D:/Corpora/upt-r50-sanity/ckpt_00000_00.pt", map_location="cpu")["model_state_dict"]
    model0 = torch.load("D:/Corpora/upt-r50-sanity/ckpt_02097_01.pt", map_location="cpu")["model_state_dict"]
    #print(model1.keys())
    #print(model1["scheduler_state_dict"]
    #model1 = model1["model_state_dict"]
    #exit(0)
    #model10 = torch.load("D:/Corpora/ckpt_20970_10.pt")["model_state_dict"]
    #model20 = torch.load("D:/Corpora/ckpt_41940_20.pt")["model_state_dict"]
    for key, param in model0.items():
        #if "detr" in key:
        if not torch.all(torch.eq(param, model1[key])).item():
            print(key)
            #print(model0[key].tolist())
            #print(model1[key].tolist())
            #exit()
                #print(model20[key].tolist())
                #print(torch.all(torch.eq(param, model20[key])))
                #print("...............")

