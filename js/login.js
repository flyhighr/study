document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
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
    
    // Handle form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get form data
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const remember = document.getElementById('remember').checked;
            
            // Validate form
            if (!email || !password) {
                showError('Please enter both email and password');
                return;
            }
            
            try {
                // Show loading state
                const submitButton = loginForm.querySelector('button[type="submit"]');
                const originalButtonText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing in...';
                
                // Make API request
                const response = await fetch('https://study-o5hp.onrender.com/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    body: `username=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    // Check if the error is about email verification
                    if (response.status === 403 && data.detail && data.detail.includes('not verified')) {
                        // Store email for verification prompt
                        localStorage.setItem('unverifiedEmail', email);
                        
                        // Redirect to verification prompt
                        window.location.href = 'verify-prompt.html';
                        return;
                    }
                    
                    throw new Error(data.detail || 'Login failed. Please check your credentials.');
                }
                
                // Login successful
                // Store token in localStorage
                localStorage.setItem('accessToken', data.access_token);
                
                // If remember me is checked, store email
                if (remember) {
                    localStorage.setItem('rememberedEmail', email);
                } else {
                    localStorage.removeItem('rememberedEmail');
                }
                
                // Fetch user data
                const userResponse = await fetch('https://study-o5hp.onrender.com/users/me', {
                    headers: {
                        'Authorization': `Bearer ${data.access_token}`
                    }
                });
                
                if (!userResponse.ok) {
                    throw new Error('Failed to fetch user data');
                }
                
                const userData = await userResponse.json();
                
                // Store user data
                localStorage.setItem('userData', JSON.stringify(userData));
                
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
    
    // Check for remembered email
    const rememberedEmail = localStorage.getItem('rememberedEmail');
    if (rememberedEmail && document.getElementById('email')) {
        document.getElementById('email').value = rememberedEmail;
        if (document.getElementById('remember')) {
            document.getElementById('remember').checked = true;
        }
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