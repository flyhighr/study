// study-sessions.js - For the study sessions page functionality

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
    
    // Load study sessions data
    loadStudySessions();
    
    // Initialize event listeners
    initEventListeners();
    
    // Check URL for new session parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('new') === 'true') {
        openAddSessionModal();
    }
    
    // Show time zone info
    showTimeZoneInfo();
    
    // Logout functionality
    document.getElementById('logout-btn').addEventListener('click', function(e) {
        e.preventDefault();
        logout();
    });
});

// Show time zone information
function showTimeZoneInfo() {
    const now = new Date();
    const timeZoneName = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const timeZoneOffset = now.getTimezoneOffset();
    const offsetHours = Math.abs(Math.floor(timeZoneOffset / 60));
    const offsetMinutes = Math.abs(timeZoneOffset % 60);
    const offsetSign = timeZoneOffset <= 0 ? '+' : '-';
    
    const offsetFormatted = `UTC${offsetSign}${offsetHours.toString().padStart(2, '0')}:${offsetMinutes.toString().padStart(2, '0')}`;
    
    console.log(`Browser time zone: ${timeZoneName} (${offsetFormatted})`);
    
    // Add time zone info to the session stats section
    const statsContainer = document.querySelector('.session-stats');
    if (statsContainer) {
        const timeZoneInfo = document.createElement('div');
        timeZoneInfo.className = 'time-zone-info';
        timeZoneInfo.style.textAlign = 'center';
        timeZoneInfo.style.marginTop = '10px';
        timeZoneInfo.style.fontSize = '12px';
        timeZoneInfo.style.color = '#666';
        timeZoneInfo.innerHTML = `<small>All times shown in your local time zone: ${timeZoneName} (${offsetFormatted})</small>`;
        statsContainer.appendChild(timeZoneInfo);
    }
}

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
        
        // Set user initial
        const userInitial = document.getElementById('user-initial');
        if (userInitial && userData.name) {
            userInitial.textContent = userData.name.charAt(0).toUpperCase();
        }
        
        // If we don't have complete user data, fetch it
        if (!userData.name || !userData.email) {
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

// Initialize event listeners
function initEventListeners() {
    // Add session button
    document.getElementById('add-session-btn').addEventListener('click', openAddSessionModal);
    document.getElementById('empty-add-btn').addEventListener('click', openAddSessionModal);
    
    // Start session button
    document.getElementById('start-session-btn').addEventListener('click', function() {
        openTimerOverlay(); // No argument means create a quick session
    });
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModals();
            }
        });
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Session form submission
    document.getElementById('session-form').addEventListener('submit', handleSessionFormSubmit);
    
    // Additional direct handler for the save button
    document.getElementById('save-btn').addEventListener('click', function(e) {
        // Only handle if it's a direct click, not a form submission
        if (e.target === this) {
            e.preventDefault();
            
            // Trigger form submission
            const form = document.getElementById('session-form');
            const submitEvent = new Event('submit', {
                bubbles: true,
                cancelable: true
            });
            form.dispatchEvent(submitEvent);
        }
    });
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteSession);
    
    // Search functionality
    document.getElementById('session-search').addEventListener('input', debounce(filterSessions, 300));
    
    // Filters
    document.getElementById('status-filter').addEventListener('change', filterSessions);
    document.getElementById('subject-filter').addEventListener('change', filterSessions);
    document.getElementById('date-filter').addEventListener('change', filterSessions);
    
    // Pomodoro toggle in form
    document.getElementById('use-pomodoro').addEventListener('change', function() {
        const pomodoroSettings = document.getElementById('pomodoro-settings');
        pomodoroSettings.style.display = this.checked ? 'block' : 'none';
        
        // Enable/disable required attribute based on toggle
        const pomodoroWork = document.getElementById('pomodoro-work');
        const pomodoroBreak = document.getElementById('pomodoro-break');
        
        if (this.checked) {
            pomodoroWork.setAttribute('required', '');
            pomodoroBreak.setAttribute('required', '');
        } else {
            pomodoroWork.removeAttribute('required');
            pomodoroBreak.removeAttribute('required');
        }
    });
    
    // Timer overlay controls - improved close button handling
    document.getElementById('close-timer-btn').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        closeTimerOverlay();
    });
    
    document.getElementById('start-timer-btn').addEventListener('click', startTimer);
    document.getElementById('pause-timer-btn').addEventListener('click', pauseTimer);
    document.getElementById('reset-timer-btn').addEventListener('click', resetTimer);
    document.getElementById('complete-session-btn').addEventListener('click', completeSession);
    
    // Pomodoro toggle in timer
    document.getElementById('pomodoro-toggle').addEventListener('change', function() {
        const pomodoroOptions = document.getElementById('pomodoro-options');
        pomodoroOptions.style.display = this.checked ? 'block' : 'none';
        
        // Reset timer when changing mode
        resetTimer();
    });
    
    // Timer options change
    document.getElementById('work-time').addEventListener('change', resetTimer);
    document.getElementById('break-time').addEventListener('change', resetTimer);
}

// Load study sessions data
async function loadStudySessions() {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading state
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'flex';
        });
        
        // Fetch subjects for the filter and form
        const subjectsResponse = await fetch('https://study-o5hp.onrender.com/subjects', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!subjectsResponse.ok) {
            throw new Error('Failed to fetch subjects');
        }
        
        const subjects = await subjectsResponse.json();
        
        // Clear existing options first to prevent duplicates
        const subjectFilter = document.getElementById('subject-filter');
        const subjectSelect = document.getElementById('subject-id');
        
        // Keep only the first option
        while (subjectFilter.options.length > 1) {
            subjectFilter.remove(1);
        }
        
        while (subjectSelect.options.length > 1) {
            subjectSelect.remove(1);
        }
        
        // Populate subject filters
        subjects.forEach(subject => {
            // Add to filter
            const filterOption = document.createElement('option');
            filterOption.value = subject._id;
            filterOption.textContent = subject.name;
            subjectFilter.appendChild(filterOption);
            
            // Add to form select
            const selectOption = document.createElement('option');
            selectOption.value = subject._id;
            selectOption.textContent = subject.name;
            subjectSelect.appendChild(selectOption);
        });
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store subjects map for later use
        window.subjectsMap = subjectsMap;
        
        // Fetch all study sessions
        const sessionsResponse = await fetch('https://study-o5hp.onrender.com/study-sessions', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!sessionsResponse.ok) {
            throw new Error('Failed to fetch study sessions');
        }
        
        const sessions = await sessionsResponse.json();
        
        // Store sessions in a global variable for filtering
        window.allSessions = sessions;
        
        // Update session statistics
        updateSessionStats(sessions);
        
        // Display sessions
        displaySessions(sessions, subjectsMap);
        
    } catch (error) {
        console.error('Error loading study sessions:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const upcomingSessionsList = document.getElementById('upcoming-sessions-list');
        upcomingSessionsList.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load study sessions. Please try again later.</p>
            </div>
        `;
    }
}

// Update session statistics
function updateSessionStats(sessions) {
    const completedSessions = sessions.filter(session => session.completed);
    const upcomingSessions = sessions.filter(session => !session.completed);
    
    // Calculate total study time in hours
    const totalStudyTime = completedSessions.reduce((total, session) => {
        return total + (session.actual_duration || 0);
    }, 0) / 60; // Convert minutes to hours
    
    // Calculate completion rate
    const completionRate = sessions.length > 0 ? Math.round((completedSessions.length / sessions.length) * 100) : 0;
    
    // Update stats in UI
    document.getElementById('total-study-time').textContent = `${totalStudyTime.toFixed(1)} hours`;
    document.getElementById('sessions-completed').textContent = completedSessions.length;
    document.getElementById('upcoming-sessions').textContent = upcomingSessions.length;
    document.getElementById('completion-rate').textContent = `${completionRate}%`;
}

// Display sessions
function displaySessions(sessions, subjectsMap) {
    const upcomingSessionsList = document.getElementById('upcoming-sessions-list');
    const pastSessionsList = document.getElementById('past-sessions-list');
    const noSessions = document.getElementById('no-sessions');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Check if there are any sessions
    if (sessions.length === 0) {
        upcomingSessionsList.innerHTML = '';
        pastSessionsList.innerHTML = '';
        noSessions.style.display = 'block';
        return;
    }
    
    // Hide no sessions message
    noSessions.style.display = 'none';
    
    // Clear existing sessions
    upcomingSessionsList.innerHTML = '';
    pastSessionsList.innerHTML = '';
    
    // Separate sessions by status
    const now = new Date();
    const upcomingSessions = sessions.filter(session => !session.completed && new Date(session.scheduled_date) >= now);
    const pastSessions = sessions.filter(session => session.completed || new Date(session.scheduled_date) < now);
    
    // Sort sessions by date
    upcomingSessions.sort((a, b) => new Date(a.scheduled_date) - new Date(b.scheduled_date));
    pastSessions.sort((a, b) => new Date(b.scheduled_date) - new Date(a.scheduled_date));
    
    // Display upcoming sessions
    if (upcomingSessions.length === 0) {
        upcomingSessionsList.innerHTML = `
            <div class="empty-state">
                <p>No upcoming study sessions scheduled</p>
            </div>
        `;
    } else {
        upcomingSessions.forEach(session => {
            const subject = subjectsMap[session.subject_id] || { name: 'No Subject', color: '#808080' };
            
            // Convert UTC date from server to local date for display
            const scheduledDate = new Date(session.scheduled_date);
            const isToday = isDateToday(scheduledDate);
            
            // Debug logging to help troubleshoot time issues
            console.log(`Session time: ${session.scheduled_date}, Local display time: ${scheduledDate}, ${formatTime(scheduledDate)}`);
            
            const sessionItem = document.createElement('div');
            sessionItem.className = `session-item ${isToday ? 'today' : ''}`;
            sessionItem.dataset.id = session._id;
            
            sessionItem.innerHTML = `
                <div class="session-header">
                    <div class="session-subject" style="background-color: ${subject.color}">
                        ${subject.name}
                    </div>
                    <div class="session-actions">
                        <button class="btn btn-sm start-session-btn" data-id="${session._id}">
                            <i class="fas fa-play"></i> Start
                        </button>
                        <button class="btn btn-icon edit-btn" data-id="${session._id}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon delete-btn" data-id="${session._id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="session-content">
                    <div class="session-time">
                        <i class="fas fa-calendar-alt"></i>
                        <span>${formatDate(scheduledDate)}</span>
                        <i class="fas fa-clock"></i>
                        <span>${formatTime(scheduledDate)}</span>
                    </div>
                    <div class="session-duration">
                        <i class="fas fa-hourglass-half"></i>
                        <span>Planned: ${session.planned_duration} minutes</span>
                    </div>
                    ${session.description ? `
                    <div class="session-description">
                        ${session.description}
                    </div>
                    ` : ''}
                    ${session.use_pomodoro ? `
                    <div class="session-pomodoro">
                        <i class="fas fa-stopwatch"></i>
                        <span>Pomodoro: ${session.pomodoro_work}min work / ${session.pomodoro_break}min break</span>
                    </div>
                    ` : ''}
                </div>
            `;
            
            upcomingSessionsList.appendChild(sessionItem);
        });
    }
    
    // Display past sessions
    if (pastSessions.length === 0) {
        pastSessionsList.innerHTML = `
            <div class="empty-state">
                <p>No past study sessions</p>
            </div>
        `;
    } else {
        pastSessions.forEach(session => {
            const subject = subjectsMap[session.subject_id] || { name: 'No Subject', color: '#808080' };
            
            // Convert UTC date from server to local date for display
            const scheduledDate = new Date(session.scheduled_date);
            const completedDate = session.completed_at ? new Date(session.completed_at) : null;
            
            const sessionItem = document.createElement('div');
            sessionItem.className = `session-item ${session.completed ? 'completed' : 'missed'}`;
            sessionItem.dataset.id = session._id;
            
            sessionItem.innerHTML = `
                <div class="session-header">
                    <div class="session-subject" style="background-color: ${subject.color}">
                        ${subject.name}
                    </div>
                    <div class="session-status">
                        ${session.completed ? 
                            `<span class="status-badge completed">Completed</span>` : 
                            `<span class="status-badge missed">Missed</span>`
                        }
                    </div>
                </div>
                <div class="session-content">
                    <div class="session-time">
                        <i class="fas fa-calendar-alt"></i>
                        <span>${formatDate(scheduledDate)}</span>
                        <i class="fas fa-clock"></i>
                        <span>${formatTime(scheduledDate)}</span>
                    </div>
                    <div class="session-duration">
                        <i class="fas fa-hourglass-half"></i>
                        <span>Planned: ${session.planned_duration} minutes</span>
                        ${session.actual_duration ? `
                        <i class="fas fa-check-circle"></i>
                        <span>Actual: ${session.actual_duration} minutes</span>
                        ` : ''}
                    </div>
                    ${session.description ? `
                    <div class="session-description">
                        ${session.description}
                    </div>
                    ` : ''}
                    ${session.completed && completedDate ? `
                    <div class="session-completed-at">
                        <i class="fas fa-check"></i>
                        <span>Completed on ${formatDate(completedDate)} at ${formatTime(completedDate)}</span>
                    </div>
                    ` : ''}
                </div>
            `;
            
            pastSessionsList.appendChild(sessionItem);
        });
    }
    
    // Add event listeners to buttons - use event delegation for better performance
    upcomingSessionsList.addEventListener('click', function(e) {
        // Find the closest button if we clicked on a child element
        const startBtn = e.target.closest('.start-session-btn');
        const editBtn = e.target.closest('.edit-btn');
        const deleteBtn = e.target.closest('.delete-btn');
        
        if (startBtn) {
            e.preventDefault();
            e.stopPropagation();
            const sessionId = startBtn.getAttribute('data-id');
            console.log("Starting session with ID:", sessionId);
            if (sessionId) {
                openTimerOverlay(sessionId);
            }
        } else if (editBtn) {
            e.preventDefault();
            e.stopPropagation();
            const sessionId = editBtn.getAttribute('data-id');
            if (sessionId) {
                openEditSessionModal(sessionId);
            }
        } else if (deleteBtn) {
            e.preventDefault();
            e.stopPropagation();
            const sessionId = deleteBtn.getAttribute('data-id');
            if (sessionId) {
                openDeleteConfirmationModal(sessionId);
            }
        }
    });
}

// Filter sessions
function filterSessions() {
    const searchTerm = document.getElementById('session-search').value.toLowerCase();
    const statusFilter = document.getElementById('status-filter').value;
    const subjectFilter = document.getElementById('subject-filter').value;
    const dateFilter = document.getElementById('date-filter').value;
    
    // Get all sessions
    const sessions = window.allSessions || [];
    
    // Apply filters
    const filteredSessions = sessions.filter(session => {
        // Search term filter
        const matchesSearch = 
            (session.description && session.description.toLowerCase().includes(searchTerm)) ||
            window.subjectsMap[session.subject_id]?.name.toLowerCase().includes(searchTerm);
        
        // Status filter
        const matchesStatus = statusFilter === 'all' || 
            (statusFilter === 'planned' && !session.completed) || 
            (statusFilter === 'completed' && session.completed);
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || session.subject_id === subjectFilter;
        
        // Date filter
        let matchesDate = true;
        const sessionDate = new Date(session.scheduled_date);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        if (dateFilter === 'today') {
            const tomorrow = new Date(today);
            tomorrow.setDate(tomorrow.getDate() + 1);
            matchesDate = sessionDate >= today && sessionDate < tomorrow;
        } else if (dateFilter === 'tomorrow') {
            const tomorrow = new Date(today);
            tomorrow.setDate(tomorrow.getDate() + 1);
            const dayAfterTomorrow = new Date(today);
            dayAfterTomorrow.setDate(today.getDate() + 2);
            matchesDate = sessionDate >= tomorrow && sessionDate < dayAfterTomorrow;
        } else if (dateFilter === 'this_week') {
            const nextWeek = new Date(today);
            nextWeek.setDate(nextWeek.getDate() + 7);
            matchesDate = sessionDate >= today && sessionDate < nextWeek;
        } else if (dateFilter === 'next_week') {
            const nextWeekStart = new Date(today);
            nextWeekStart.setDate(nextWeekStart.getDate() + 7);
            const nextWeekEnd = new Date(nextWeekStart);
            nextWeekEnd.setDate(nextWeekEnd.getDate() + 7);
            matchesDate = sessionDate >= nextWeekStart && sessionDate < nextWeekEnd;
        }
        
        return matchesSearch && matchesStatus && matchesSubject && matchesDate;
    });
    
    // Display filtered sessions
    displaySessions(filteredSessions, window.subjectsMap);
}

// Open add session modal
function openAddSessionModal() {
    // Reset form
    document.getElementById('session-form').reset();
    document.getElementById('session-id').value = '';
    
    // Hide pomodoro settings
    document.getElementById('pomodoro-settings').style.display = 'none';
    
    // Set default scheduled date to current time + 1 hour (properly handling timezone)
    const defaultDate = new Date();
    defaultDate.setHours(defaultDate.getHours() + 1);
    document.getElementById('scheduled-date').value = formatDateTimeForInput(defaultDate);
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Plan Study Session';
    document.getElementById('save-btn').textContent = 'Save Session';
    
    // Show modal
    document.getElementById('session-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit session modal
function openEditSessionModal(sessionId) {
    if (!sessionId) {
        console.error('No session ID provided for editing');
        return;
    }
    
    const token = localStorage.getItem('accessToken');
    
    // Show loading in the modal
    document.getElementById('modal-title').textContent = 'Loading Session...';
    document.getElementById('session-modal').classList.add('active');
    document.body.classList.add('modal-open');
    
    // Fetch session details
    fetch(`https://study-o5hp.onrender.com/study-sessions/${sessionId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load session details');
        }
        return response.json();
    })
    .then(session => {
        // Populate form
        document.getElementById('session-id').value = session._id;
        document.getElementById('subject-id').value = session.subject_id;
        
        // Format date properly for local timezone
        const scheduledDate = new Date(session.scheduled_date);
        document.getElementById('scheduled-date').value = formatDateTimeForInput(scheduledDate);
        
        document.getElementById('planned-duration').value = session.planned_duration;
        document.getElementById('description').value = session.description || '';
        document.getElementById('use-pomodoro').checked = session.use_pomodoro;
        
        // Show/hide pomodoro settings
        document.getElementById('pomodoro-settings').style.display = session.use_pomodoro ? 'block' : 'none';
        
        // Set pomodoro values
        document.getElementById('pomodoro-work').value = session.pomodoro_work || 25;
        document.getElementById('pomodoro-break').value = session.pomodoro_break || 5;
        
        // Update modal title
        document.getElementById('modal-title').textContent = 'Edit Study Session';
        document.getElementById('save-btn').textContent = 'Update Session';
    })
    .catch(error => {
        console.error('Error fetching session details:', error);
        alert('Failed to load session details. Please try again.');
        closeModals();
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(sessionId) {
    if (!sessionId) {
        console.error('No session ID provided for deletion');
        return;
    }
    
    // Store session ID for deletion
    document.getElementById('confirm-delete-btn').dataset.id = sessionId;
    
    // Show modal
    document.getElementById('delete-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open timer overlay
function openTimerOverlay(sessionId) {
    // If sessionId is not provided or is an event object, open quick session modal
    if (!sessionId || sessionId instanceof Event || typeof sessionId === 'object') {
        console.log("Starting a quick session");
        openQuickSessionModal();
        return;
    }
    
    // Otherwise, load the specified session
    const token = localStorage.getItem('accessToken');
    
    // Show loading in timer
    document.getElementById('timer-subject').querySelector('.subject-name').textContent = 'Loading...';
    document.getElementById('timer-overlay').classList.add('active');
    document.body.classList.add('overlay-open');
    
    fetch(`https://study-o5hp.onrender.com/study-sessions/${sessionId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load session details');
        }
        return response.json();
    })
    .then(session => {
        // Store session data
        window.currentSession = session;
        
        // Set up timer with session data
        setupTimer(session);
    })
    .catch(error => {
        console.error('Error fetching session details:', error);
        alert('Failed to load session details. Please try again.');
        closeTimerOverlay();
    });
}

// Open quick session modal
function openQuickSessionModal() {
    console.log("Opening quick session modal");
    
    // First close any open modals
    closeModals();
    
    // Open session modal with current time
    openAddSessionModal();
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Start Quick Study Session';
    document.getElementById('save-btn').textContent = 'Start Session';
    
    // Set a flag to indicate we should open timer after saving
    window.openTimerAfterSave = true;
    
    // Make sure the form is clear and properly set up
    const form = document.getElementById('session-form');
    form.reset();
    
    // Set default scheduled date to now
    const now = new Date();
    document.getElementById('scheduled-date').value = formatDateTimeForInput(now);
    
    // Focus on the subject dropdown
    setTimeout(() => {
        try {
            const subjectSelect = document.getElementById('subject-id');
            subjectSelect.focus();
            
            // If there's only one subject (plus the default option), select it automatically
            if (subjectSelect.options.length === 2) {
                subjectSelect.selectedIndex = 1;
            }
        } catch (e) {
            console.error("Error focusing on subject select:", e);
        }
    }, 100);
}

// Setup timer with session data
function setupTimer(session) {
    const subject = window.subjectsMap[session.subject_id] || { name: 'Study Session', color: '#4287f5' };
    
    // Set subject info
    const subjectElement = document.getElementById('timer-subject');
    subjectElement.querySelector('.subject-name').textContent = subject.name;
    subjectElement.querySelector('.subject-color').style.backgroundColor = subject.color;
    
    // Set pomodoro toggle
    document.getElementById('pomodoro-toggle').checked = session.use_pomodoro;
    document.getElementById('pomodoro-options').style.display = session.use_pomodoro ? 'block' : 'none';
    
    // Set pomodoro values
    document.getElementById('work-time').value = session.pomodoro_work || 25;
    document.getElementById('break-time').value = session.pomodoro_break || 5;
    
    // Initialize timer
    resetTimer();
    
    // Initialize elapsed time
    window.elapsedSeconds = 0;
    if (window.elapsedInterval) {
        clearInterval(window.elapsedInterval);
    }
    window.elapsedInterval = null;
    document.getElementById('elapsed-time').textContent = '00:00:00';
    
    // Initialize elapsed time
    window.elapsedSeconds = 0;
    if (window.elapsedInterval) {
        clearInterval(window.elapsedInterval);
    }
    window.elapsedInterval = null;
    document.getElementById('elapsed-time').textContent = '00:00:00';
    
    // Reset pomodoro cycles
    window.pomodoroCycles = 0;
    document.getElementById('pomodoro-cycles').textContent = '0';
    
    // Reset buttons
    document.getElementById('start-timer-btn').disabled = false;
    document.getElementById('pause-timer-btn').disabled = true;
    document.getElementById('reset-timer-btn').disabled = true;
    
    // Reset completion flags
    window.isClosingTimer = false;
    window.hasAskedToSave = false;
}

// Close timer overlay
function closeTimerOverlay() {
    // Prevent multiple executions
    if (window.isClosingTimer) return;
    window.isClosingTimer = true;
    
    // Stop any running timers
    if (window.timerInterval) {
        clearInterval(window.timerInterval);
        window.timerInterval = null;
    }
    
    if (window.elapsedInterval) {
        clearInterval(window.elapsedInterval);
        window.elapsedInterval = null;
    }
    
    // Check if we need to save progress
    const shouldAskToSave = window.elapsedSeconds > 0 && 
                           window.currentSession && 
                           !window.currentSession.completed &&
                           !window.hasAskedToSave; // New flag to prevent multiple prompts
    
    // Set flag to prevent multiple prompts
    window.hasAskedToSave = true;
    
    // Hide overlay
    document.getElementById('timer-overlay').classList.remove('active');
    document.body.classList.remove('overlay-open');
    
    // Ask if user wants to mark the session as completed
    if (shouldAskToSave) {
        if (confirm('Do you want to save your progress for this study session?')) {
            completeSession();
        }
    }
    
    // Reset the closing flag after a short delay
    setTimeout(() => {
        window.isClosingTimer = false;
        window.hasAskedToSave = false;
    }, 500);
}

// Start timer
function startTimer() {
    // Disable start button, enable pause and reset
    document.getElementById('start-timer-btn').disabled = true;
    document.getElementById('pause-timer-btn').disabled = false;
    document.getElementById('reset-timer-btn').disabled = false;
    
    // Get timer values
    const isPomodoroMode = document.getElementById('pomodoro-toggle').checked;
    const workMinutes = parseInt(document.getElementById('work-time').value) || 25;
    const breakMinutes = parseInt(document.getElementById('break-time').value) || 5;
    
    // Set initial timer state
    if (!window.timerState) {
        window.timerState = {
            isBreak: false,
            timeLeft: workMinutes * 60,
            totalTime: workMinutes * 60
        };
    }
    
    // Update timer display
    updateTimerDisplay();
    
    // Start timer interval
    if (window.timerInterval) {
        clearInterval(window.timerInterval);
    }
    
    window.timerInterval = setInterval(() => {
        // Decrement time left
        window.timerState.timeLeft--;
        
        // Update display
        updateTimerDisplay();
        
        // Check if timer is done
        if (window.timerState.timeLeft <= 0) {
            // Play sound
            playTimerEndSound();
            
            if (isPomodoroMode) {
                // Toggle between work and break
                window.timerState.isBreak = !window.timerState.isBreak;
                
                // If switching from break to work, increment cycle count
                if (!window.timerState.isBreak) {
                    window.pomodoroCycles++;
                    document.getElementById('pomodoro-cycles').textContent = window.pomodoroCycles;
                }
                
                // Set new time based on mode
                if (window.timerState.isBreak) {
                    window.timerState.timeLeft = breakMinutes * 60;
                    window.timerState.totalTime = breakMinutes * 60;
                    document.getElementById('timer-status').textContent = 'Break Time';
                    document.getElementById('timer-status').className = 'timer-status break';
                } else {
                    window.timerState.timeLeft = workMinutes * 60;
                    window.timerState.totalTime = workMinutes * 60;
                    document.getElementById('timer-status').textContent = 'Work Time';
                    document.getElementById('timer-status').className = 'timer-status work';
                }
                
                // Update display
                updateTimerDisplay();
            } else {
                // Non-pomodoro mode, stop timer
                clearInterval(window.timerInterval);
                window.timerInterval = null;
                
                // Reset buttons
                document.getElementById('start-timer-btn').disabled = false;
                document.getElementById('pause-timer-btn').disabled = true;
                
                // Notify user
                alert('Timer finished! You can start again or complete the session.');
            }
        }
    }, 1000);
    
    // Start elapsed time counter if not already running
    if (!window.elapsedInterval) {
        window.elapsedInterval = setInterval(() => {
            window.elapsedSeconds++;
            updateElapsedTime();
        }, 1000);
    }
}

// Pause timer
function pauseTimer() {
    // Stop timer
    if (window.timerInterval) {
        clearInterval(window.timerInterval);
        window.timerInterval = null;
    }
    
    // Stop elapsed time counter
    if (window.elapsedInterval) {
        clearInterval(window.elapsedInterval);
        window.elapsedInterval = null;
    }
    
    // Update buttons
    document.getElementById('start-timer-btn').disabled = false;
    document.getElementById('pause-timer-btn').disabled = true;
}

// Reset timer
function resetTimer() {
    // Stop timer
    if (window.timerInterval) {
        clearInterval(window.timerInterval);
        window.timerInterval = null;
    }
    
    // Get timer values
    const isPomodoroMode = document.getElementById('pomodoro-toggle').checked;
    const workMinutes = parseInt(document.getElementById('work-time').value) || 25;
    
    // Reset timer state
    window.timerState = {
        isBreak: false,
        timeLeft: workMinutes * 60,
        totalTime: workMinutes * 60
    };
    
    // Reset status
    document.getElementById('timer-status').textContent = 'Work Time';
    document.getElementById('timer-status').className = 'timer-status work';
    
    // Update display
    updateTimerDisplay();
    
    // Reset buttons
    document.getElementById('start-timer-btn').disabled = false;
    document.getElementById('pause-timer-btn').disabled = true;
    document.getElementById('reset-timer-btn').disabled = true;
}

// Update timer display
function updateTimerDisplay() {
    if (!window.timerState) return;
    
    const minutes = Math.floor(window.timerState.timeLeft / 60);
    const seconds = window.timerState.timeLeft % 60;
    
    // Update time display
    document.getElementById('timer-display').textContent = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    // Update progress bar
    const progressPercent = 100 - Math.floor((window.timerState.timeLeft / window.timerState.totalTime) * 100);
    document.getElementById('timer-progress').style.width = `${progressPercent}%`;
}

// Update elapsed time display
function updateElapsedTime() {
    if (typeof window.elapsedSeconds !== 'number') {
        window.elapsedSeconds = 0;
    }
    
    const hours = Math.floor(window.elapsedSeconds / 3600);
    const minutes = Math.floor((window.elapsedSeconds % 3600) / 60);
    const seconds = window.elapsedSeconds % 60;
    
    document.getElementById('elapsed-time').textContent = 
        `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// Play timer end sound
function playTimerEndSound() {
    try {
        // Create a simple beep sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.type = 'sine';
        oscillator.frequency.value = 800;
        gainNode.gain.value = 0.5;
        
        oscillator.start();
        
        // Stop after 500ms
        setTimeout(() => {
            oscillator.stop();
        }, 500);
    } catch (error) {
        console.log('Audio alert not supported or blocked by browser');
        // Fallback to console log
        console.log('Timer ended - BEEP!');
    }
}

// Complete session
function completeSession() {
    // Set flag to prevent completion prompt when closing
    window.hasAskedToSave = true;
    
    const token = localStorage.getItem('accessToken');
    const session = window.currentSession;
    
    if (!session) {
        alert('No active session to complete');
        closeTimerOverlay();
        return;
    }
    
    // Disable the complete button to prevent double submissions
    const completeBtn = document.getElementById('complete-session-btn');
    completeBtn.disabled = true;
    completeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    // Prepare update data
    const updateData = {
        completed: true,
        completed_at: new Date().toISOString()
    };
    
    // Add actual duration if we have elapsed time
    if (window.elapsedSeconds) {
        updateData.actual_duration = Math.max(1, Math.ceil(window.elapsedSeconds / 60)); // At least 1 minute
    } else {
        // If no elapsed time, use planned duration
        updateData.actual_duration = session.planned_duration;
    }
    
    // Update session
    fetch(`https://study-o5hp.onrender.com/study-sessions/${session._id}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(updateData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.detail || 'Failed to update session');
            });
        }
        return response.json();
    })
    .then(updatedSession => {
        // Update the session in our local data
        if (window.allSessions) {
            window.allSessions = window.allSessions.map(s => 
                s._id === session._id ? updatedSession : s
            );
        }
        
        // Close overlay without triggering another completion prompt
        document.getElementById('timer-overlay').classList.remove('active');
        document.body.classList.remove('overlay-open');
        
        // Update the UI without full reload
        if (window.subjectsMap) {
            displaySessions(window.allSessions, window.subjectsMap);
            updateSessionStats(window.allSessions);
        } else {
            // Fallback to full reload
            loadStudySessions();
        }
        
        // Show success message
        alert('Study session completed successfully!');
    })
    .catch(error => {
        console.error('Error completing session:', error);
        alert(`Failed to complete session: ${error.message}`);
        
        // Reset button state
        completeBtn.disabled = false;
        completeBtn.textContent = 'Complete Session';
    });
}

// Close all modals
function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    document.body.classList.remove('modal-open');
}

// Handle session form submission
async function handleSessionFormSubmit(e) {
    e.preventDefault();
    console.log("Form submitted");
    
    const token = localStorage.getItem('accessToken');
    const sessionId = document.getElementById('session-id').value;
    const isEdit = sessionId !== '';
    
    // Get form data
    const formData = {
        subject_id: document.getElementById('subject-id').value,
        planned_duration: parseInt(document.getElementById('planned-duration').value) || 60,
        description: document.getElementById('description').value || '',
        use_pomodoro: document.getElementById('use-pomodoro').checked
    };
    
    // Handle the scheduled date with proper timezone awareness
    const scheduledDateInput = document.getElementById('scheduled-date').value;
    if (scheduledDateInput) {
        // Convert to ISO string but preserve the user's intended local time
        const localDate = new Date(scheduledDateInput);
        formData.scheduled_date = localDate.toISOString();
        
        console.log(`Local input time: ${scheduledDateInput}, Sending to server: ${formData.scheduled_date}`);
    } else {
        alert('Please select a scheduled date');
        return;
    }
    
    // Add pomodoro settings if enabled
    if (formData.use_pomodoro) {
        formData.pomodoro_work = parseInt(document.getElementById('pomodoro-work').value) || 25;
        formData.pomodoro_break = parseInt(document.getElementById('pomodoro-break').value) || 5;
    } else {
        // Set default values when not using pomodoro
        formData.pomodoro_work = 25;
        formData.pomodoro_break = 5;
    }
    
    // Show loading state
    const saveBtn = document.getElementById('save-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    console.log("Sending form data:", formData);
    
    try {
        let response;
        
        if (isEdit) {
            // Update existing session
            response = await fetch(`https://study-o5hp.onrender.com/study-sessions/${sessionId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new session
            response = await fetch('https://study-o5hp.onrender.com/study-sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        }
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error("API error response:", errorData);
            throw new Error(errorData.detail || 'Failed to save session');
        }
        
        const savedSession = await response.json();
        console.log("Session saved successfully:", savedSession);
        
        // Update our local data
        if (isEdit && window.allSessions) {
            window.allSessions = window.allSessions.map(s => 
                s._id === sessionId ? savedSession : s
            );
        } else if (window.allSessions) {
            window.allSessions.push(savedSession);
        }
        
        // Close modal
        closeModals();
        
        // Check if we should open timer immediately
        const shouldOpenTimer = window.openTimerAfterSave;
        window.openTimerAfterSave = false;
        
        if (shouldOpenTimer) {
            console.log("Opening timer for new session");
            window.currentSession = savedSession;
            openTimerOverlay(savedSession._id);
        } else {
            // Update the UI without full reload
            if (window.subjectsMap) {
                displaySessions(window.allSessions, window.subjectsMap);
                updateSessionStats(window.allSessions);
            } else {
                // Fallback to full reload
                loadStudySessions();
            }
        }
        
    } catch (error) {
        console.error('Error saving session:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

// Delete session
async function deleteSession() {
    const token = localStorage.getItem('accessToken');
    const sessionId = document.getElementById('confirm-delete-btn').dataset.id;
    
    if (!sessionId) {
        console.error('No session ID found for deletion');
        alert('Error: Could not identify which session to delete.');
        closeModals();
        return;
    }
    
    // Show loading state
    const deleteBtn = document.getElementById('confirm-delete-btn');
    const originalBtnText = deleteBtn.textContent;
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
    
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/study-sessions/${sessionId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete session');
        }
        
        // Update our local data
        if (window.allSessions) {
            window.allSessions = window.allSessions.filter(s => s._id !== sessionId);
        }
        
        // Close modal
        closeModals();
        
        // Update the UI without full reload
        if (window.subjectsMap) {
            displaySessions(window.allSessions, window.subjectsMap);
            updateSessionStats(window.allSessions);
        } else {
            // Fallback to full reload
            loadStudySessions();
        }
        
    } catch (error) {
        console.error('Error deleting session:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        deleteBtn.disabled = false;
        deleteBtn.textContent = originalBtnText;
        
        // Clear the session ID
        deleteBtn.dataset.id = '';
    }
}

// Helper functions for date/time handling
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatTime(date) {
    const options = { hour: 'numeric', minute: '2-digit', hour12: true };
    return date.toLocaleTimeString('en-US', options);
}

function formatDateTimeForInput(date) {
    // Ensure we're working with a valid date
    if (!(date instanceof Date) || isNaN(date)) {
        date = new Date();
    }
    
    // Format as YYYY-MM-DDTHH:MM in local time zone
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function isDateToday(date) {
    const today = new Date();
    return date.getDate() === today.getDate() &&
           date.getMonth() === today.getMonth() &&
           date.getFullYear() === today.getFullYear();
}

// Debounce function for search input
function debounce(func, delay) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), delay);
    };
}

// Logout function
function logout() {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('userData');
    window.location.href = 'login.html';
}