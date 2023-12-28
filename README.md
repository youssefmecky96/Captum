#changes to captum package 
file:attr/_core/integrated_gradients.py  line 360
reason: input from pytorch forecasting such as group id has no bearing on output and had to account for cases when gradient was none 
change:scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads if grad is not None
        ]
file:attr/_core/integrated_gradients.py  line 383
reason: input from pytorch forecasting such as group id has no bearing on output and had to account for cases when gradient was none 
change:elif (len(total_grads) == 2) or (len(total_grads) == 3):
            inputs = inputs[1]
            baselines = baselines[1]
            total_grads = total_grads[0]
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
            )
file:attr/_core/integrated_gradients.py  line 359
reason: input from pytorch forecasting such as group id has no bearing on output and had to account for cases when gradient was none 
change:grads=[
            grad
            for grad in grads if grad is not None
        ]
file:attr/_core/gradient.py  line 119
reason: input from pytorch forecasting such as group id has no bearing on output and had to account for cases when gradient was none 
change:grads = torch.autograd.grad(torch.unbind(outputs), inputs,allow_unused=True)
file : attr/_core/saliency.py
reason: get gradient of only active part of the input from pytorch forecasting
change: gradients = gradients[1]
file: attr/_core/feature_ablation.py line 570,line 413
reason: moving tensors to GPU 
change: ablated_tensor = (
            expanded_input *
            (1 - current_mask).to(expanded_input.dtype).to(expanded_input.device)
        ) + (baseline * current_mask.to(expanded_input.dtype).to(expanded_input.device))
        return ablated_tensor, current_mask
file: attr/_core/feature_ablation.py line line 413
reason: moving tensors to GPU 
change:  total_attrib[i] += (eval_diff * current_mask.to(attrib_type).to(eval_diff.device)).sum(
file: _utils/gradient.py line 122
reason:force gradient computation even when model is in eval mode
change: torch.set_grad_enabled(True)
        



