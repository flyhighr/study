document.addEventListener('DOMContentLoaded', function() {
    // Get token from URL
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    const resetLoading = document.querySelector('.reset-password-loading');
    const resetForm = document.querySelector('.reset-password-form');
    const resetSuccess = document.querySelector('.reset-password-success');
    const resetError = document.querySelector('.reset-password-error');
    const errorMessage = document.getElementById('error-message');
    const resetPasswordForm = document.getElementById('reset-password-form');
    const togglePasswordBtn = document.querySelector('.toggle-password');
    const passwordInput = document.getElementById('password');
    
    // Store token in hidden field
    if (token && document.getElementById('reset-token')) {
        document.getElementById('reset-token').value = token;
    }
    
    // Toggle password visibility
    if (togglePasswordBtn && passwordInput) {
        togglePasswordBtn.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            this.classList.toggle('fa-eye');
            this.classList.toggle('fa-eye-slash');
        });
    }
    
    // Password strength checker
    if (passwordInput) {
        passwordInput.addEventListener('input', checkPasswordStrength);
    }
    
    // Validate token
    if (token) {
        // Show form - in a real app, you might validate the token first
        resetLoading.style.display = 'none';
        resetForm.style.display = 'block';
    } else {
        // No token provided
        resetLoading.style.display = 'none';
        resetError.style.display = 'block';
    }
    
    // Handle form submission
    if (resetPasswordForm) {
        resetPasswordForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const token = document.getElementById('reset-token').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            
            // Validate form
            if (!password || !confirmPassword) {
                showError('Please fill in all required fields');
                return;
            }
            
            if (password !== confirmPassword) {
                showError('Passwords do not match');
                return;
            }
            
            // Validate password strength
            if (!isPasswordStrong(password)) {
                showError('Please ensure your password meets all requirements');
                return;
            }
            
            try {
                // Show loading state
                const submitButton = resetPasswordForm.querySelector('button[type="submit"]');
                const originalButtonText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resetting...';
                
                // Make API request
                const response = await fetch('https://study-o5hp.onrender.com/reset-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        token,
                        new_password: password
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Password reset failed. Please try again.');
                }
                
                // Reset successful
                resetForm.style.display = 'none';
                resetSuccess.style.display = 'block';
                
            } catch (error) {
                // Reset button
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
                
                // Show error
                showError(error.message);
            }
        });
    }
    
    // Function to check password strength
    function checkPasswordStrength() {
        const password = passwordInput.value;
        const strengthMeter = document.querySelector('.strength-meter-fill');
        
        // Check requirements
        const hasLength = password.length >= 8;
        const hasUppercase = /[A-Z]/.test(password);
        const hasLowercase = /[a-z]/.test(password);
        const hasNumber = /[0-9]/.test(password);
        
        // Update requirement indicators
        updateRequirement('length', hasLength);
        updateRequirement('uppercase', hasUppercase);
        updateRequirement('lowercase', hasLowercase);
        updateRequirement('number', hasNumber);
        
        // Calculate strength
        let strength = 0;
        if (hasLength) strength += 25;
        if (hasUppercase) strength += 25;
        if (hasLowercase) strength += 25;
        if (hasNumber) strength += 25;
        
        // Update strength meter
        if (strengthMeter) {
            strengthMeter.style.width = `${strength}%`;
            strengthMeter.setAttribute('data-strength', Math.floor(strength / 25));
        }
    }
    
    // Function to update requirement indicator
    function updateRequirement(requirement, isMet) {
        const requirementElement = document.querySelector(`[data-requirement="${requirement}"]`);
        if (requirementElement) {
            const icon = requirementElement.querySelector('i');
            if (isMet) {
                icon.className = 'fas fa-check';
                requirementElement.classList.add('met');
            } else {
                icon.className = 'fas fa-times';
                requirementElement.classList.remove('met');
            }
        }
    }
    
    // Function to check if password is strong
    function isPasswordStrong(password) {
        return (
            password.length >= 8 &&
            /[A-Z]/.test(password) &&
            /[a-z]/.test(password) &&
            /[0-9]/.test(password)
        );
    }
    
    // Function to show error message
    function showError(message) {
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            
            // Hide error after 5 seconds
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
    }
});