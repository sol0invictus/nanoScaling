
import math
import torch
from torch.optim import AdamW

class StructuredAdamW(AdamW):
    """
    Implements AdamW algorithm with Structured Weight Decay (Group Lasso style).
    
    Inherits from torch.optim.AdamW.
    
    Instead of standard L2 weight decay (which pushes all weights uniformly towards zero),
    this optimizer applies decay based on the L2 norm of entire rows or columns.
    
    Decay term: lambda * (p / ||p||_group)
    Update: p -= lr * lambda * (p / ||p||_group)
    
    This acts as a "decoupled" weight decay step performed before the standard Adam optimization step.
    To allow this, we initialize the base AdamW with weight_decay=0.0 and handle decay manually.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, group_mode='row', epsilon_decay=1e-5, **kwargs):
        
        # Filter kwargs to only include valid AdamW arguments if needed, 
        # but generally we just pass params, lr, betas, eps, amsgrad, weight_decay=0.
        # We handle 'structured_weight_decay' ourselves.
        
        # We must set weight_decay=0 for the base AdamW to avoid applying standard L2 decay.
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=0.0, amsgrad=amsgrad)
        
        # We need to initialize the optimizer with these properties so they are part ofparam_groups.
        # However, AdamW.__init__ will validate keys in defaults and might reject unknown ones if strict?
        # torch.optim.Optimizer loads defaults into param_groups. It DOES NOT validate keys usually.
        # But AdamW.__init__ might pass **defaults to super? No, it usually does:
        # super(AdamW, self).__init__(params, defaults)
        # So passing extra keys in 'defaults' is usually FINE for Optimizer class.
        # Wait, the error was "TypeError: AdamW.__init__() got an unexpected keyword argument 'structured_weight_decay'"
        # indicated that I was passing it to super().__init__(params, **defaults) <-- unpacked!
        # Ah! I was doing: super(StructuredAdamW, self).__init__(params, **defaults)
        
        # Standard Optimizer usage: super().__init__(params, defaults) <-- passes dict as second arg!
        # But AdamW usage: super(AdamW, self).__init__(params, defaults)
        # Wait, let's look at AdamW definition.
        # def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None):
        #    ...
        #    super().__init__(params, defaults)
        
        # So if I call super(StructuredAdamW, self).__init__(params, **defaults) I am calling AdamW.__init__ with named args.
        # If 'defaults' has 'structured_weight_decay', AdamW.__init__ explodes because it doesn't take that arg.
        
        # Solution: Call AdamW.__init__ explicitly with known args, OR rely on positional args?
        # Better: pass known args to super().__init__ explicitly or filtered.
        # And then Update param_groups with my custom values.
        
        super(StructuredAdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, 
                                              weight_decay=0.0, amsgrad=amsgrad, **kwargs)
        
        # Now inject our custom values into param_groups
        for group in self.param_groups:
            group['structured_weight_decay'] = weight_decay
            group['group_mode'] = group_mode
            group['epsilon_decay'] = epsilon_decay
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Retrieve custom parameters
            weight_decay = group.get('structured_weight_decay', 0.0)
            group_mode = group.get('group_mode', 'row')
            epsilon_decay = group.get('epsilon_decay', 1e-5)
            lr = group['lr']
            
            # Apply Structured Weight Decay (Decoupled)
            if weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                        
                    if p.dim() >= 2:
                        # Apply structured decay
                        norm = None
                        if group_mode == 'row':
                            norm = p.norm(p=2, dim=1, keepdim=True)
                        elif group_mode == 'col':
                            norm = p.norm(p=2, dim=0, keepdim=True)
                        
                        if norm is not None:
                             # Decoupled decay: p = p - lr * lambda * (p / (norm + eps))
                             scaled_p = p / (norm + epsilon_decay)
                             p.data.add_(scaled_p, alpha=-lr * weight_decay)
                        else:
                             # Fallback or 'none' mode
                             p.data.mul_(1 - lr * weight_decay)
                             
                    else:
                        # 1D params (biases etc) -> Standard L2 decay
                        p.data.mul_(1 - lr * weight_decay)

        # Proceed with standard AdamW step (with internal weight_decay=0)
        super(StructuredAdamW, self).step(closure)

        return loss
