import unittest
import torch
import torch.nn.functional as F
from training.direction_loss import direction_weights,weighted_answer_loss
class CharTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
class DirectionLossTests(unittest.TestCase):
    def test_ones_matches_standard_and_masks_prompt(self):
        torch.manual_seed(1);x=torch.randn(2,5,9,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4],[-100,-100,1,2,-100]])
        got=weighted_answer_loss(x,labels,torch.ones_like(labels,dtype=torch.float))
        expected=F.cross_entropy(x[:,:-1].reshape(-1,9),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(got,expected);got.backward()
        self.assertEqual(x.grad[:,0].abs().sum().item(),0)
        self.assertEqual(x.grad[:,-1].abs().sum().item(),0)
    def test_weighted_hand_calculation(self):
        x=torch.tensor([[[2.,0.],[0.,2.],[1.,1.]]],requires_grad=True)
        labels=torch.tensor([[-100,0,0]]);w=torch.tensor([[1.,1.,6.]])
        loss=weighted_answer_loss(x,labels,w)
        pieces=F.cross_entropy(x[0,:2],torch.tensor([0,0]),reduction='none')
        torch.testing.assert_close(loss,(pieces[0]+6*pieces[1])/7)
    def test_only_direction_not_shared_suffix_or_question(self):
        for answer in ['人在狗的左边。','人在狗的右边。']:
            batch={'labels':torch.tensor([[-100,-100]+[ord(c) for c in answer]]),'answer_starts':[2]}
            rows=[{'task':'spatial','answer':answer}]
            w=direction_weights(batch,rows,CharTokenizer(),6)
            self.assertEqual((w==6).sum().item(),1)
            self.assertEqual(w[0,2+answer.index('边')].item(),1)
            rows[0]['task']='listing';self.assertTrue((direction_weights(batch,rows,CharTokenizer())==1).all())
    def test_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            direction_weights({'labels':torch.tensor([[-100,1,2]]),'answer_starts':[1]},[{'task':'spatial','answer':'左边'}],CharTokenizer())
if __name__=='__main__':unittest.main()
