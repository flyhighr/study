// register.js - For the registration page functionality

document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('register-form');
    const errorMessage = document.getElementById('error-message');
    const togglePasswordBtn = document.querySelector('.toggle-password');
    const passwordInput = document.getElementById('password');
    
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
    
    // Handle form submission
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get form data
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const termsChecked = document.getElementById('terms').checked;
            
            // Validate form
            if (!name || !email || !password) {
                showError('Please fill in all required fields');
                return;
            }
            
            if (!termsChecked) {
                showError('You must agree to the Terms of Service and Privacy Policy');
                return;
            }
            
            // Validate password strength
            if (!isPasswordStrong(password)) {
                showError('Please ensure your password meets all requirements');
                return;
            }
            
            try {
                // Show loading state
                const submitButton = registerForm.querySelector('button[type="submit"]');
                const originalButtonText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
                
                // Make API request
                const response = await fetch('https://study-o5hp.onrender.com/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name,
                        email,
                        password
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Registration failed. Please try again.');
                }
                
                // Registration successful, now login
                const loginResponse = await fetch('https://study-o5hp.onrender.com/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    body: `username=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
                });
                
                const loginData = await loginResponse.json();
                
                if (!loginResponse.ok) {
                    throw new Error('Account created but login failed. Please go to login page.');
                }
                
                // Store token and user data
                localStorage.setItem('accessToken', loginData.access_token);
                localStorage.setItem('userData', JSON.stringify(data));
                
                // Redirect to dashboard
                window.location.href = 'dashboard.html';
                
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