import unittest
import torch
from eventfieldnet.evidence_supervision.objectives import counterfactual_evidence_loss
from eventfieldnet.evidence_supervision.pairing import reciprocal_pairs

class EvidenceTests(unittest.TestCase):
    def test_four_corners_match_explicit_pair_formula_and_gradients(self):
        a=torch.tensor([[.4,.2],[.1,-.2]],requires_grad=True)
        b=torch.tensor([[.3,.1],[-.1,.2]],requires_grad=True)
        w=torch.ones(2,1,2)/2
        loss,_=counterfactual_evidence_loss(a,b,w,torch.tensor([1,0]),torch.ones(2,dtype=torch.bool),arm="C02")
        x=a.mean(1);y=b.mean(1)
        expected=(torch.relu(.2-x[0]+y[0])+torch.relu(.2-x[0]+y[1])+torch.relu(.2-x[1]+y[1])+torch.relu(.2-x[1]+y[0]))/4
        self.assertTrue(torch.equal(loss,expected))
        g=torch.autograd.grad(loss,(a,b),retain_graph=True)
        h=torch.autograd.grad(expected,(a,b))
        for v,u in zip(g,h):self.assertTrue(torch.equal(v,u))
    def test_empty_pair_returns_connected_zero(self):
        a=torch.randn(2,3,requires_grad=True);b=torch.randn(2,3,requires_grad=True)
        loss,probe=counterfactual_evidence_loss(a,b,torch.ones(2,1,3)/3,torch.tensor([0,1]),torch.zeros(2,dtype=torch.bool),arm="C02")
        self.assertEqual(loss.item(),0);loss.backward()
        self.assertEqual(a.grad.abs().sum().item(),0)
        self.assertEqual(probe["active_queries"].item(),0)
    def test_reciprocal_pair_excludes_same_video(self):
        rows=[dict(qid=1,vid="a",query="running man"),dict(qid=2,vid="a",query="red car"),dict(qid=3,vid="b",query="blue boat")]
        pairs,valid,_=reciprocal_pairs(rows,"cpu")
        self.assertEqual(pairs.tolist(),[2,1,0]);self.assertEqual(valid.tolist(),[True,False,True])
    def test_other_arms_rejected(self):
        with self.assertRaises(ValueError):
            counterfactual_evidence_loss(torch.zeros(2,1),torch.zeros(2,1),torch.ones(2,1,1),torch.tensor([1,0]),torch.ones(2,dtype=torch.bool),arm="C03")
