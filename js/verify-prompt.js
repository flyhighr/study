document.addEventListener('DOMContentLoaded', function() {
    const userEmail = document.getElementById('user-email');
    const resendBtn = document.getElementById('resend-verification');
    const logoutBtn = document.getElementById('logout-btn');
    const resendStatus = document.getElementById('resend-status');
    
    // Check if user is logged in
    const userData = localStorage.getItem('userData');
    if (!userData) {
        window.location.href = 'login.html';
        return;
    }
    
    const user = JSON.parse(userData);
    
    // Display user's email
    if (userEmail) {
        userEmail.textContent = user.email;
    }
    
    // Handle resend verification
    if (resendBtn) {
        resendBtn.addEventListener('click', async function() {
            try {
                resendBtn.disabled = true;
                resendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
                
                const response = await fetch('https://study-o5hp.onrender.com/resend-verification', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email: user.email })
                });
                
                const data = await response.json();
                
                // Show success message
                if (resendStatus) {
                    resendStatus.className = 'resend-status success';
                    resendStatus.textContent = 'Verification email sent! Please check your inbox.';
                    resendStatus.style.display = 'block';
                }
                
                // Reset button
                setTimeout(() => {
                    resendBtn.disabled = false;
                    resendBtn.innerHTML = 'Resend Verification Email';
                }, 3000);
                
            } catch (error) {
                console.error('Error:', error);
                
                // Show error message
                if (resendStatus) {
                    resendStatus.className = 'resend-status error';
                    resendStatus.textContent = 'Failed to send verification email. Please try again.';
                    resendStatus.style.display = 'block';
                }
                
                // Reset button
                resendBtn.disabled = false;
                resendBtn.innerHTML = 'Resend Verification Email';
            }
        });
    }
    
    // Handle logout
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function() {
            localStorage.removeItem('accessToken');
            localStorage.removeItem('userData');
            window.location.href = 'login.html';
        });
    }
});