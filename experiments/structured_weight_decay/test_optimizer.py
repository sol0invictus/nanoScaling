
import torch
import unittest
import sys
import os

# Generic path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir))) # root
sys.path.append(current_dir) # experiment dir

from optimizer import StructuredAdamW

class TestStructuredAdamW(unittest.TestCase):
    def test_structured_step_row(self):
        # Test row-wise decay
        p = torch.tensor([[10.0, 0.0], [1.0, 0.0]], requires_grad=True)
        
        # Initialize with weight_decay which maps to structured_weight_decay internally
        optim = StructuredAdamW([p], lr=0.1, weight_decay=1.0, group_mode='row', betas=(0.0, 0.0), eps=1e-8)
        
        # Zero grad to ignore Adam update part (or momentum)
        p.grad = torch.zeros_like(p)
        
        optim.step()
        
        # Check values
        # Row 1: 10 -> 9.9
        self.assertAlmostEqual(p[0,0].item(), 9.9, places=5)
        # Row 2: 1 -> 0.9
        self.assertAlmostEqual(p[1,0].item(), 0.9, places=5)
        
    def test_structured_vs_standard(self):
        # Compare standard AdamW behavior (approx) vs Structured
        # Standard AdamW (Group Mode 'none')
        # Note: If mode is none, our implementation falls back to p.data.mul_(1 - lr * wd) which is approximation of AdamW
        # The base AdamW usually does p -= lr * wd * p + ... 
        # But we set base weight_decay=0. So we are solely responsible.
        p_std = torch.tensor([[10.0, 0.0]], requires_grad=True)
        # Use weight_decay arg
        optim_std = StructuredAdamW([p_std], lr=0.1, weight_decay=1.0, group_mode='none')
        p_std.grad = torch.zeros_like(p_std)
        optim_std.step()
        
        # Standard decay: p *= (1 - 0.1*1.0) = 0.9*p
        self.assertAlmostEqual(p_std[0,0].item(), 9.0, places=5)
        
        # Structured Row
        p_row = torch.tensor([[10.0, 0.0]], requires_grad=True)
        optim_row = StructuredAdamW([p_row], lr=0.1, weight_decay=1.0, group_mode='row')
        p_row.grad = torch.zeros_like(p_row)
        optim_row.step()
        
        # Structured: p -= 0.1 * p/norm = 10 - 0.1 * 10/10 = 9.9
        self.assertAlmostEqual(p_row[0,0].item(), 9.9, places=5)
        
    def test_col_mode(self):
        # Col 1: [3, 4] -> norm 5.
        p = torch.tensor([[3.0], [4.0]], requires_grad=True)
        optim = StructuredAdamW([p], lr=0.1, weight_decay=1.0, group_mode='col')
        p.grad = torch.zeros_like(p)
        optim.step()
        
        # Expected: p -= 0.1 * p / 5.
        # p[0] = 3 - 0.1 * 3/5 = 3 - 0.06 = 2.94
        # p[1] = 4 - 0.1 * 4/5 = 4 - 0.08 = 3.92
        self.assertAlmostEqual(p[0,0].item(), 2.94, places=5)
        self.assertAlmostEqual(p[1,0].item(), 3.92, places=5)

if __name__ == '__main__':
    unittest.main()
