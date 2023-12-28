# Changes to Captum Package

This document outlines the changes made to the Captum package, specifically to the `integrated_gradients.py`, `gradient.py`, `saliency.py`, and `feature_ablation.py` files. The primary focus of these changes is to handle inputs from PyTorch forecasting, such as group ID, which has no bearing on the output, and to account for cases when the gradient is None.

### Changes in `integrated_gradients.py`

**File:** `attr/_core/integrated_gradients.py`

1. **Line 360**
   - **Reason:** Handling cases when the gradient is None.
   - **Change:**
     ```python
     scaled_grads = [
         grad.contiguous().view(n_steps, -1) * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
         for grad in grads if grad is not None
     ]
     ```

2. **Line 383**
   - **Reason:** Handling cases when the gradient is None.
   - **Change:**
     ```python
     elif (len(total_grads) == 2) or (len(total_grads) == 3):
         inputs = inputs[1]
         baselines = baselines[1]
         total_grads = total_grads[0]
         attributions = tuple(
             total_grad * (input - baseline)
             for total_grad, input, baseline in zip(total_grads, inputs, baselines)
         )
     ```

3. **Line 359**
   - **Reason:** Handling cases when the gradient is None.
   - **Change:**
     ```python
     grads=[
         grad
         for grad in grads if grad is not None
     ]
     ```

### Changes in `gradient.py`

**File:** `attr/_core/gradient.py`

1. **Line 119**
   - **Reason:** Handling inputs from PyTorch forecasting with no bearing on output.
   - **Change:**
     ```python
     grads = torch.autograd.grad(torch.unbind(outputs), inputs, allow_unused=True)
     ```

### Changes in `saliency.py`

**File:** `attr/_core/saliency.py`

1. **Reason:** To get the gradient of only the active part of the input from PyTorch forecasting.
   - **Change:**
     ```python
     gradients = gradients[1]
     ```

### Changes in `feature_ablation.py`

**File:** `attr/_core/feature_ablation.py`

1. **Lines 570, 413**
   - **Reason:** Moving tensors to GPU.
   - **Change:**
     ```python
     ablated_tensor = (
         expanded_input * (1 - current_mask).to(expanded_input.dtype).to(expanded_input.device)
     ) + (baseline * current_mask.to(expanded_input.dtype).to(expanded_input.device))
     return ablated_tensor, current_mask
     ```
   
2. **Line 413**
   - **Reason:** Moving tensors to GPU.
   - **Change:**
     ```python
     total_attrib[i] += (eval_diff * current_mask.to(attrib_type).to(eval_diff.device)).sum()
     ```
