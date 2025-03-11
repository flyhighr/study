document.addEventListener('DOMContentLoaded', function() {
    const forgotPasswordForm = document.getElementById('forgot-password-form');
    const errorMessage = document.getElementById('error-message');
    const formContainer = document.querySelector('.forgot-password-form');
    const successContainer = document.querySelector('.forgot-password-success');
    
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            
            if (!email) {
                showError('Please enter your email address');
                return;
            }
            
            try {
                // Show loading state
                const submitButton = forgotPasswordForm.querySelector('button[type="submit"]');
                const originalButtonText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
                
                // Make API request
                const response = await fetch('https://study-o5hp.onrender.com/forgot-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email })
                });
                
                const data = await response.json();
                
                // Hide form, show success message
                formContainer.style.display = 'none';
                successContainer.style.display = 'block';
                
            } catch (error) {
                // Reset button
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
                
                // Show error
                showError('An error occurred. Please try again.');
            }
        });
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