document.addEventListener('DOMContentLoaded', function() {
    // Check if user is logged in
    const token = localStorage.getItem('accessToken');
    if (!token) {
        window.location.href = 'login.html';
        return;
    }

    // Initialize sidebar functionality
    initSidebar();
    
    // Load user data
    loadUserData();
    
    // Load dashboard data
    loadDashboardData();
    
    // Set greeting based on time of day
    setGreeting();
    
    // Initialize event listeners
    initEventListeners();
    
    // Fetch timezones for the profile edit form
    fetchTimezones();
});

// Initialize sidebar functionality
function initSidebar() {
    const toggleSidebarBtn = document.querySelector('.toggle-sidebar');
    const closeSidebarBtn = document.querySelector('.close-sidebar');
    const sidebar = document.querySelector('.sidebar');
    
    if (toggleSidebarBtn && sidebar) {
        toggleSidebarBtn.addEventListener('click', function() {
            sidebar.classList.toggle('active');
            document.body.classList.toggle('sidebar-open');
        });
    }
    
    if (closeSidebarBtn && sidebar) {
        closeSidebarBtn.addEventListener('click', function() {
            sidebar.classList.remove('active');
            document.body.classList.remove('sidebar-open');
        });
    }
}

// Load user data
async function loadUserData() {
    try {
        const token = localStorage.getItem('accessToken');
        const userData = JSON.parse(localStorage.getItem('userData') || '{}');
        
        // Update user info in sidebar
        document.getElementById('user-name').textContent = userData.name || 'User';
        document.getElementById('user-email').textContent = userData.email || 'user@example.com';
        document.getElementById('greeting-name').textContent = userData.name?.split(' ')[0] || 'Student';
        
        // Set user timezone
        if (userData.timezone) {
            document.getElementById('user-timezone').innerHTML = `<i class="fas fa-globe"></i> ${userData.timezone}`;
        }
        
        // Set user initial
        const userInitial = document.getElementById('user-initial');
        if (userInitial && userData.name) {
            userInitial.textContent = userData.name.charAt(0).toUpperCase();
        }
        
        // If we don't have complete user data, fetch it
        if (!userData.name || !userData.email || !userData.timezone) {
            const response = await fetch('https://study-o5hp.onrender.com/users/me', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user data');
            }
            
            const freshUserData = await response.json();
            localStorage.setItem('userData', JSON.stringify(freshUserData));
            
            // Update UI with fresh data
            document.getElementById('user-name').textContent = freshUserData.name;
            document.getElementById('user-email').textContent = freshUserData.email;
            document.getElementById('greeting-name').textContent = freshUserData.name?.split(' ')[0] || 'Student';
            document.getElementById('user-timezone').innerHTML = `<i class="fas fa-globe"></i> ${freshUserData.timezone || 'UTC'}`;
            
            if (userInitial) {
                userInitial.textContent = freshUserData.name.charAt(0).toUpperCase();
            }
        }
    } catch (error) {
        console.error('Error loading user data:', error);
        // If we can't load user data, redirect to login
        if (error.message === 'Failed to fetch user data') {
            logout();
        }
    }
}

// Fetch available timezones from the API
async function fetchTimezones() {
    try {
        const token = localStorage.getItem('accessToken');
        const response = await fetch('https://study-o5hp.onrender.com/timezones', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch timezones');
        }
        
        const data = await response.json();
        const timezones = data.timezones;
        
        // Populate timezone dropdown
        const timezoneSelect = document.getElementById('profile-timezone');
        timezoneSelect.innerHTML = '';
        
        // Add browser's detected timezone at the top
        const browserTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        const optionBrowser = document.createElement('option');
        optionBrowser.value = browserTimezone;
        optionBrowser.textContent = `${browserTimezone} (Browser Detected)`;
        timezoneSelect.appendChild(optionBrowser);
        
        // Add separator
        const optionSeparator = document.createElement('option');
        optionSeparator.disabled = true;
        optionSeparator.textContent = '──────────';
        timezoneSelect.appendChild(optionSeparator);
        
        // Add all available timezones
        timezones.forEach(tz => {
            const option = document.createElement('option');
            option.value = tz;
            option.textContent = tz;
            timezoneSelect.appendChild(option);
        });
        
        // Set the current user's timezone as selected
        const userData = JSON.parse(localStorage.getItem('userData') || '{}');
        if (userData.timezone) {
            timezoneSelect.value = userData.timezone;
        } else {
            timezoneSelect.value = browserTimezone;
        }
        
    } catch (error) {
        console.error('Error fetching timezones:', error);
    }
}

// Handle profile form submission
async function handleProfileUpdate(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const nameInput = document.getElementById('profile-name');
    const emailInput = document.getElementById('profile-email');
    const timezoneSelect = document.getElementById('profile-timezone');
    
    // Show loading state
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        const updateData = {
            name: nameInput.value,
            email: emailInput.value,
            timezone: timezoneSelect.value
        };
        
        const response = await fetch('https://study-o5hp.onrender.com/users/me', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(updateData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to update profile');
        }
        
        const updatedUser = await response.json();
        
        // Update local storage
        localStorage.setItem('userData', JSON.stringify(updatedUser));
        
        // Update UI
        document.getElementById('user-name').textContent = updatedUser.name;
        document.getElementById('user-email').textContent = updatedUser.email;
        document.getElementById('greeting-name').textContent = updatedUser.name?.split(' ')[0] || 'Student';
        document.getElementById('user-timezone').innerHTML = `<i class="fas fa-globe"></i> ${updatedUser.timezone || 'UTC'}`;
        document.getElementById('user-initial').textContent = updatedUser.name.charAt(0).toUpperCase();
        
        // Close modal
        closeModals();
        
        // Show success message
        alert('Profile updated successfully!');
        
    } catch (error) {
        console.error('Error updating profile:', error);
        alert(error.message);
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
}

// Initialize event listeners
function initEventListeners() {
    // Logout button
    document.getElementById('logout-btn').addEventListener('click', function(e) {
        e.preventDefault();
        logout();
    });
    
    // Edit profile button
    document.getElementById('edit-profile-btn').addEventListener('click', openProfileModal);
    
    // Timezone badge (also can open profile modal)
    document.getElementById('user-timezone').addEventListener('click', openProfileModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel profile edit button
    document.getElementById('cancel-profile-btn').addEventListener('click', closeModals);
    
    // Profile form submission
    document.getElementById('profile-form').addEventListener('submit', handleProfileUpdate);
    
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModals();
            }
        });
    });
}

// Open profile modal
function openProfileModal() {
    const userData = JSON.parse(localStorage.getItem('userData') || '{}');
    
    // Populate form with current user data
    document.getElementById('profile-name').value = userData.name || '';
    document.getElementById('profile-email').value = userData.email || '';
    
    const timezoneSelect = document.getElementById('profile-timezone');
    if (userData.timezone && timezoneSelect.querySelector(`option[value="${userData.timezone}"]`)) {
        timezoneSelect.value = userData.timezone;
    }
    
    // Show modal
    document.getElementById('profile-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Close all modals
function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    document.body.classList.remove('modal-open');
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading spinners
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'flex';
        });
        
        // Fetch statistics
        const statsResponse = await fetch('https://study-o5hp.onrender.com/statistics', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!statsResponse.ok) {
            throw new Error('Failed to fetch statistics');
        }
        
        const statsData = await statsResponse.json();
        
        // Update summary cards
        document.getElementById('pending-assignments-count').textContent = statsData.pending_assignments;
        document.getElementById('upcoming-events-count').textContent = statsData.upcoming_events;
        document.getElementById('study-hours').textContent = statsData.study_hours.toFixed(1) + " hours";
        
        const completionRate = statsData.total_assignments > 0 
            ? Math.round((statsData.completed_assignments / statsData.total_assignments) * 100) 
            : 0;
        document.getElementById('completion-rate').textContent = completionRate + "%";
        
        // Render subject progress chart
        renderSubjectProgressChart(statsData.subject_stats);
        
        // Render study time chart
        renderStudyTimeChart(statsData.daily_study_data);
        
        // Fetch upcoming assignments
        const assignmentsResponse = await fetch('https://study-o5hp.onrender.com/assignments?status=pending&limit=5', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!assignmentsResponse.ok) {
            throw new Error('Failed to fetch assignments');
        }
        
        const assignments = await assignmentsResponse.json();
        
        // Display upcoming assignments
        displayUpcomingAssignments(assignments);
        
        // Fetch upcoming events
        const eventsResponse = await fetch('https://study-o5hp.onrender.com/events?limit=5&start_after=' + new Date().toISOString(), {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!eventsResponse.ok) {
            throw new Error('Failed to fetch events');
        }
        
        const events = await eventsResponse.json();
        
        // Display upcoming events
        displayUpcomingEvents(events);
        
        // Fetch recent materials
        const materialsResponse = await fetch('https://study-o5hp.onrender.com/materials?limit=5', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!materialsResponse.ok) {
            throw new Error('Failed to fetch materials');
        }
        
        const materials = await materialsResponse.json();
        
        // Display recent materials
        displayRecentMaterials(materials);
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        
        // Hide loading spinners
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message in each section
        const errorHtml = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load data. Please try again later.</p>
            </div>
        `;
        
        document.getElementById('upcoming-assignments-list').innerHTML = errorHtml;
        document.getElementById('upcoming-events-list').innerHTML = errorHtml;
        document.getElementById('recent-materials-list').innerHTML = errorHtml;
    }
}

// Display upcoming assignments
function displayUpcomingAssignments(assignments) {
    const container = document.getElementById('upcoming-assignments-list');
    container.innerHTML = '';
    
    if (assignments.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-tasks"></i>
                <p>No upcoming assignments</p>
            </div>
        `;
        return;
    }
    
    assignments.sort((a, b) => new Date(a.due_date) - new Date(b.due_date));
    
    assignments.forEach(assignment => {
        const dueDate = new Date(assignment.due_date);
        const isOverdue = dueDate < new Date() && assignment.status !== 'completed';
        
        const taskItem = document.createElement('div');
        taskItem.className = 'task-item';
        
        taskItem.innerHTML = `
            <div class="task-details">
                <h4>${assignment.title}</h4>
                <p class="task-due-date ${isOverdue ? 'overdue' : ''}">
                    <i class="fas fa-calendar-alt"></i> 
                    ${formatDate(dueDate)}
                </p>
            </div>
            <div class="task-priority ${assignment.priority}">
                ${capitalizeFirstLetter(assignment.priority)}
            </div>
        `;
        
        container.appendChild(taskItem);
    });
}

// Display upcoming events
function displayUpcomingEvents(events) {
    const container = document.getElementById('upcoming-events-list');
    container.innerHTML = '';
    
    if (events.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-calendar-day"></i>
                <p>No upcoming events</p>
            </div>
        `;
        return;
    }
    
    events.sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
    
    events.forEach(event => {
        const startTime = new Date(event.start_time);
        
        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        
        eventItem.innerHTML = `
            <div class="event-time">
                <span class="event-day">${startTime.getDate()}</span>
                <span class="event-month">${startTime.toLocaleString('default', { month: 'short' })}</span>
            </div>
            <div class="event-details">
                <h4>${event.title}</h4>
                <p>
                    <i class="fas fa-clock"></i> 
                    ${formatTime(startTime)}
                </p>
            </div>
            <div class="event-type ${event.type}">
                ${capitalizeFirstLetter(event.type)}
            </div>
        `;
        
        container.appendChild(eventItem);
    });
}

// Display recent materials
function displayRecentMaterials(materials) {
    const container = document.getElementById('recent-materials-list');
    container.innerHTML = '';
    
    if (materials.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-file-alt"></i>
                <p>No materials uploaded yet</p>
            </div>
        `;
        return;
    }
    
    materials.forEach(material => {
        const uploadDate = new Date(material.uploaded_at);
        
        const materialItem = document.createElement('div');
        materialItem.className = 'material-item';
        
        let fileIcon;
        switch (material.file_type) {
            case 'pdf':
                fileIcon = 'fa-file-pdf';
                break;
            case 'image':
                fileIcon = 'fa-file-image';
                break;
            case 'spreadsheet':
                fileIcon = 'fa-file-excel';
                break;
            default:
                fileIcon = 'fa-file-alt';
        }
        
        materialItem.innerHTML = `
            <div class="material-icon">
                <i class="fas ${fileIcon}"></i>
            </div>
            <div class="material-details">
                <h4>${material.name}</h4>
                <p>
                    <i class="fas fa-calendar-alt"></i> 
                    ${formatDate(uploadDate)}
                </p>
            </div>
            <a href="https://study-o5hp.onrender.com/materials/download/${material._id}?token=${localStorage.getItem('accessToken')}" 
               class="material-download" target="_blank">
                <i class="fas fa-download"></i>
            </a>
        `;
        
        container.appendChild(materialItem);
    });
}

// Render subject progress chart
function renderSubjectProgressChart(subjectStats) {
    const ctx = document.getElementById('subject-progress-chart').getContext('2d');
    
    if (subjectStats.length === 0) {
        // No data to display
        document.querySelector('.subject-progress .card-body').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-pie"></i>
                <p>No subject data available</p>
            </div>
        `;
        return;
    }
    
    const labels = subjectStats.map(subject => subject.subject_name);
    const completionData = subjectStats.map(subject => subject.completion_percentage);
    const backgroundColors = subjectStats.map(subject => subject.color);
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: completionData,
                backgroundColor: backgroundColors,
                borderColor: 'rgba(255, 255, 255, 0.8)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#333',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        },
                        padding: 20,
                        usePointStyle: true,
                        boxWidth: 10
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Poppins', sans-serif",
                        size: 14
                    },
                    bodyFont: {
                        family: "'Poppins', sans-serif",
                        size: 12
                    },
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${Math.round(context.raw)}% completed`;
                        }
                    }
                }
            },
            cutout: '70%'
        }
    });
}

// Render study time chart
function renderStudyTimeChart(dailyStudyData) {
    const ctx = document.getElementById('study-time-chart').getContext('2d');
    
    if (!dailyStudyData || dailyStudyData.length === 0) {
        // No data to display
        document.querySelector('.study-time-chart .card-body').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-bar"></i>
                <p>No study time data available</p>
            </div>
        `;
        return;
    }
    
    const labels = dailyStudyData.map(day => day.date);
    const studyHours = dailyStudyData.map(day => day.hours);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Study Hours',
                data: studyHours,
                backgroundColor: 'rgba(74, 108, 247, 0.7)',
                borderColor: 'rgba(74, 108, 247, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Hours',
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    ticks: {
                        font: {
                            family: "'Poppins', sans-serif",
                            size: 12
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleFont: {
                        family: "'Poppins', sans-serif",
                        size: 14
                    },
                    bodyFont: {
                        family: "'Poppins', sans-serif",
                        size: 12
                    },
                    callbacks: {
                        label: function(context) {
                            return `${context.raw.toFixed(1)} hours`;
                        }
                    }
                }
            }
        }
    });
}

// Set greeting based on time of day
function setGreeting() {
    const hour = new Date().getHours();
    const timeOfDayElement = document.getElementById('time-of-day');
    
    if (hour < 12) {
        timeOfDayElement.textContent = 'morning';
    } else if (hour < 18) {
        timeOfDayElement.textContent = 'afternoon';
    } else {
        timeOfDayElement.textContent = 'evening';
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatTime(date) {
    const options = { hour: 'numeric', minute: '2-digit', hour12: true };
    return date.toLocaleTimeString('en-US', options);
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Logout function
function logout() {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('userData');
    window.location.href = 'login.html';
}