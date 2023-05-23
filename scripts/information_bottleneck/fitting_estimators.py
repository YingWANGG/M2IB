import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_estim(feature1, feature2,eps=1):
    bs, n = feature1.shape
    es = torch.zeros(bs).to(device)
    for i in range(bs):
        x1, x2 = feature1[i,:], feature2[i,:]
        sig11 = torch.outer(x1, x1)
        sig22 = torch.outer(x2, x2)
        concat_emb = torch.concat([x1, x2])
        sig = torch.outer(concat_emb, concat_emb)
        det1 = torch.det(sig11+eps*torch.eye(n).to(device))
        det2 = torch.det(sig22+eps*torch.eye(n).to(device))
        det = torch.det(sig+eps*torch.eye(2*n).to(device))
        es[i] = 0.5*torch.log(det1*det2/det)
    return es

def pearson_estim(feature1, feature2):
    cos = torch.nn.CosineSimilarity(eps=1e-6)
    rho = cos(feature1 - feature1.mean(dim=1, keepdim=True), feature2 - feature2.mean(dim=1, keepdim=True))
    es =  -0.5 * torch.log(1 - rho*rho)
    return es

def cos_estim(feature1, feature2):
    cos = torch.nn.CosineSimilarity(eps=1e-6)
    return cos(feature1, feature2)
