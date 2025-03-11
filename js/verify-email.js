document.addEventListener('DOMContentLoaded', function() {
    // Get token from URL
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    const verificationLoading = document.querySelector('.verification-loading');
    const verificationSuccess = document.querySelector('.verification-success');
    const verificationError = document.querySelector('.verification-error');
    const resendForm = document.querySelector('.resend-form');
    const resendSuccess = document.querySelector('.resend-success');
    
    // Handle resend button click
    const resendBtn = document.getElementById('resend-btn');
    if (resendBtn) {
        resendBtn.addEventListener('click', function() {
            verificationError.style.display = 'none';
            resendForm.style.display = 'block';
        });
    }
    
    // Handle resend form submission
    const submitResendBtn = document.getElementById('submit-resend');
    if (submitResendBtn) {
        submitResendBtn.addEventListener('click', async function() {
            const email = document.getElementById('email').value;
            
            if (!email) {
                alert('Please enter your email address');
                return;
            }
            
            try {
                submitResendBtn.disabled = true;
                submitResendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
                
                const response = await fetch('https://study-o5hp.onrender.com/resend-verification', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email })
                });
                
                const data = await response.json();
                
                resendForm.style.display = 'none';
                resendSuccess.style.display = 'block';
                
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
                
                submitResendBtn.disabled = false;
                submitResendBtn.innerHTML = 'Send Verification Link';
            }
        });
    }
    
    // If token exists, verify it
    if (token) {
        verifyToken(token);
    } else {
        // No token provided, show resend form
        verificationLoading.style.display = 'none';
        resendForm.style.display = 'block';
    }
    
    async function verifyToken(token) {
        try {
            const response = await fetch('https://study-o5hp.onrender.com/verify-email', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ token })
            });
            
            const data = await response.json();
            
            // Hide loading
            verificationLoading.style.display = 'none';
            
            if (response.ok) {
                // Show success
                verificationSuccess.style.display = 'block';
            } else {
                // Show error
                const errorMessage = document.getElementById('error-message');
                if (errorMessage) {
                    errorMessage.textContent = data.detail || 'Verification failed. Please try again.';
                }
                verificationError.style.display = 'block';
            }
            
        } catch (error) {
            console.error('Error:', error);
            
            // Hide loading, show error
            verificationLoading.style.display = 'none';
            const errorMessage = document.getElementById('error-message');
            if (errorMessage) {
                errorMessage.textContent = 'An error occurred. Please try again.';
            }
            verificationError.style.display = 'block';
        }
    }
});