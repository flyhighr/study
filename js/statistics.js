// statistics.js - For the statistics page functionality with REAL data only

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
    
    // Initialize date range selector
    initDateRangeSelector();
    
    // Load statistics data
    loadStatistics();
    
    // Initialize event listeners
    initEventListeners();
    
    // Logout functionality
    document.getElementById('logout-btn').addEventListener('click', function(e) {
        e.preventDefault();
        logout();
    });
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

// Initialize date range selector
function initDateRangeSelector() {
    const dateRangeSelect = document.getElementById('date-range');
    const customDateRange = document.getElementById('custom-date-range');
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    
    // Set default date range (last 30 days)
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    startDateInput.value = formatDateForInput(thirtyDaysAgo);
    endDateInput.value = formatDateForInput(today);
    
    // Store current date range in window object
    window.currentDateRange = {
        startDate: thirtyDaysAgo,
        endDate: today
    };
    
    // Handle date range selection change
    dateRangeSelect.addEventListener('change', function() {
        const value = this.value;
        
        if (value === 'custom') {
            customDateRange.style.display = 'block';
        } else {
            customDateRange.style.display = 'none';
            
            // Calculate date range based on selection
            const today = new Date();
            let startDate = new Date();
            
            switch (value) {
                case '7':
                    startDate.setDate(today.getDate() - 7);
                    break;
                case '14':
                    startDate.setDate(today.getDate() - 14);
                    break;
                case '30':
                    startDate.setDate(today.getDate() - 30);
                    break;
                case '90':
                    startDate.setDate(today.getDate() - 90);
                    break;
                case '180':
                    startDate.setDate(today.getDate() - 180);
                    break;
                case '365':
                    startDate.setDate(today.getDate() - 365);
                    break;
            }
            
            // Update inputs
            startDateInput.value = formatDateForInput(startDate);
            endDateInput.value = formatDateForInput(today);
            
            // Update current date range
            window.currentDateRange = {
                startDate: startDate,
                endDate: today
            };
            
            // Reload statistics with new date range
            loadStatistics();
        }
    });
    
    // Handle apply button for custom date range
    document.getElementById('apply-date-range').addEventListener('click', function() {
        const startDate = new Date(startDateInput.value);
        const endDate = new Date(endDateInput.value);
        
        // Validate date range
        if (startDate > endDate) {
            alert('Start date must be before end date');
            return;
        }
        
        // Update current date range
        window.currentDateRange = {
            startDate: startDate,
            endDate: endDate
        };
        
        // Reload statistics with new date range
        loadStatistics();
    });
}

// Initialize event listeners
function initEventListeners() {
    // Export button
    document.getElementById('export-btn').addEventListener('click', openExportModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Export cancel button
    document.getElementById('export-cancel-btn').addEventListener('click', closeModals);
    
    // Export confirm button
    document.getElementById('export-confirm-btn').addEventListener('click', handleExport);
}

// Load ALL statistics data from backend
async function loadStatistics() {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading state
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'flex';
        });
        
        // Get date range
        const { startDate, endDate } = window.currentDateRange;
        
        // Fetch statistics from backend API
        const statsUrl = `https://study-o5hp.onrender.com/statistics?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}`;
        
        const statsResponse = await fetch(statsUrl, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!statsResponse.ok) {
            throw new Error('Failed to fetch statistics');
        }
        
        const statsData = await statsResponse.json();
        
        // Fetch additional data needed for complete statistics
        const [studySessions, goals, events] = await Promise.all([
            fetchStudySessions(token, startDate, endDate),
            fetchGoals(token, startDate, endDate),
            fetchEvents(token, startDate, endDate)
        ]);
        
        // Update summary cards with real data
        updateSummaryCards(statsData, studySessions, goals, events);
        
        // Create charts with real data
        createSubjectTimeChart(statsData.subject_stats);
        createAssignmentCompletionChart(statsData);
        createDailyStudyChart(studySessions);
        createSubjectPerformanceChart(statsData.subject_stats);
        
        // Display subject statistics with real data
        displaySubjectStats(statsData.subject_stats);
        
        // Hide loading spinners
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
    } catch (error) {
        console.error('Error loading statistics:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const subjectStats = document.getElementById('subject-stats');
        subjectStats.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load statistics. Please try again later.</p>
            </div>
        `;
    }
}

// Fetch study sessions for the given date range
async function fetchStudySessions(token, startDate, endDate) {
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/study-sessions?scheduled_after=${startDate.toISOString()}&scheduled_before=${endDate.toISOString()}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch study sessions');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error fetching study sessions:', error);
        return [];
    }
}

// Fetch goals for the given date range
async function fetchGoals(token, startDate, endDate) {
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/goals`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch goals');
        }
        
        const goals = await response.json();
        
        // Filter goals within the date range
        return goals.filter(goal => {
            const createdAt = new Date(goal.created_at);
            return createdAt >= startDate && createdAt <= endDate;
        });
    } catch (error) {
        console.error('Error fetching goals:', error);
        return [];
    }
}

// Fetch events for the given date range
async function fetchEvents(token, startDate, endDate) {
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/events?start_after=${startDate.toISOString()}&start_before=${endDate.toISOString()}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch events');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error fetching events:', error);
        return [];
    }
}

// Update summary cards with real data
function updateSummaryCards(statsData, studySessions, goals, events) {
    // Assignment statistics
    document.getElementById('total-assignments').textContent = statsData.total_assignments;
    document.getElementById('completed-assignments').textContent = statsData.completed_assignments;
    document.getElementById('pending-assignments').textContent = statsData.pending_assignments;
    
    // Event statistics - calculate from real data
    const now = new Date();
    const upcomingEvents = events.filter(event => new Date(event.start_time) > now).length;
    const pastEvents = events.filter(event => new Date(event.start_time) <= now).length;
    const totalEvents = events.length;
    
    document.getElementById('total-events').textContent = totalEvents;
    document.getElementById('upcoming-events').textContent = upcomingEvents;
    document.getElementById('past-events').textContent = pastEvents;
    
    // Study time statistics
    document.getElementById('total-study-hours').textContent = `${statsData.study_hours.toFixed(1)}h`;
    
    // Calculate average daily study hours from real data
    const { startDate, endDate } = window.currentDateRange;
    const daysDiff = Math.max(1, Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)));
    const avgDailyStudy = statsData.study_hours / daysDiff;
    document.getElementById('avg-daily-study').textContent = `${avgDailyStudy.toFixed(1)}h`;
    
    // Calculate average session duration from real data
    const completedSessions = studySessions.filter(session => session.completed);
    let avgSessionDuration = 0;
    
    if (completedSessions.length > 0) {
        const totalDuration = completedSessions.reduce((sum, session) => sum + (session.actual_duration || 0), 0);
        avgSessionDuration = totalDuration / completedSessions.length;
    }
    
    // Format avg session duration (convert minutes to hours and minutes if needed)
    let avgSessionText;
    if (avgSessionDuration >= 60) {
        const hours = Math.floor(avgSessionDuration / 60);
        const minutes = Math.round(avgSessionDuration % 60);
        avgSessionText = `${hours}h ${minutes}m`;
    } else {
        avgSessionText = `${Math.round(avgSessionDuration)}m`;
    }
    
    document.getElementById('avg-session-duration').textContent = avgSessionText;
    
    // Goal statistics from real data
    const totalGoals = goals.length;
    const completedGoals = goals.filter(goal => goal.completed).length;
    const completionRate = totalGoals > 0 ? Math.round((completedGoals / totalGoals) * 100) : 0;
    
    document.getElementById('total-goals').textContent = totalGoals;
    document.getElementById('completed-goals').textContent = completedGoals;
    document.getElementById('goals-completion-rate').textContent = `${completionRate}%`;
}

// Create subject time chart with real data
function createSubjectTimeChart(subjectStats) {
    const ctx = document.getElementById('subject-time-chart').getContext('2d');
    
    // Clear any existing chart
    if (window.subjectTimeChart) {
        window.subjectTimeChart.destroy();
    }
    
    // Remove any existing empty state message
    const emptyMessage = ctx.canvas.parentNode.querySelector('.empty-chart-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Reset canvas display
    ctx.canvas.style.display = 'block';
    
    // If no data, show empty state
    if (!subjectStats || subjectStats.length === 0) {
        ctx.canvas.style.display = 'none';
        ctx.canvas.parentNode.innerHTML += '<div class="empty-chart-message">No data available for the selected time period</div>';
        return;
    }
    
    // Prepare data
    const labels = subjectStats.map(subject => subject.subject_name);
    const studyHours = subjectStats.map(subject => subject.study_hours);
    const colors = subjectStats.map(subject => subject.color || getRandomColor());
    
    // Create chart
    window.subjectTimeChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: studyHours,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                            return `${context.label}: ${value.toFixed(1)}h (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create assignment completion chart with real data
function createAssignmentCompletionChart(statsData) {
    const ctx = document.getElementById('assignment-completion-chart').getContext('2d');
    
    // Clear any existing chart
    if (window.assignmentCompletionChart) {
        window.assignmentCompletionChart.destroy();
    }
    
    // Remove any existing empty state message
    const emptyMessage = ctx.canvas.parentNode.querySelector('.empty-chart-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Reset canvas display
    ctx.canvas.style.display = 'block';
    
    // Prepare data
    const completed = statsData.completed_assignments;
    const pending = statsData.pending_assignments;
    
    // If no data, show empty state
    if (completed === 0 && pending === 0) {
        ctx.canvas.style.display = 'none';
        ctx.canvas.parentNode.innerHTML += '<div class="empty-chart-message">No assignments available for the selected time period</div>';
        return;
    }
    
    // Create chart
    window.assignmentCompletionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Completed', 'Pending'],
            datasets: [{
                data: [completed, pending],
                backgroundColor: ['#4CAF50', '#FFC107'],
                borderColor: ['#4CAF50', '#FFC107'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                            return `${context.label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create daily study chart with real data from study sessions
function createDailyStudyChart(studySessions) {
    const ctx = document.getElementById('daily-study-chart').getContext('2d');
    
    // Clear any existing chart
    if (window.dailyStudyChart) {
        window.dailyStudyChart.destroy();
    }
    
    // Remove any existing empty state message
    const emptyMessage = ctx.canvas.parentNode.querySelector('.empty-chart-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Reset canvas display
    ctx.canvas.style.display = 'block';
    
    // Generate last 7 days dates
    const days = [];
    const dailyHours = [];
    const today = new Date();
    
    for (let i = 0; i < 7; i++) {
        const date = new Date(today);
        date.setDate(today.getDate() - (6 - i));
        days.push(formatDayName(date));
        
        // Set hours to 0 for start of day
        date.setHours(0, 0, 0, 0);
        
        // Get next day
        const nextDate = new Date(date);
        nextDate.setDate(date.getDate() + 1);
        
        // Filter sessions for this day
        const daysSessions = studySessions.filter(session => {
            const sessionDate = new Date(session.completed_at || session.scheduled_date);
            return sessionDate >= date && sessionDate < nextDate && session.completed;
        });
        
        // Calculate total hours
        const totalMinutes = daysSessions.reduce((total, session) => total + (session.actual_duration || 0), 0);
        const totalHours = totalMinutes / 60;
        
        dailyHours.push(totalHours);
    }
    
    // If no data, show empty state
    if (dailyHours.every(hours => hours === 0)) {
        ctx.canvas.style.display = 'none';
        ctx.canvas.parentNode.innerHTML += '<div class="empty-chart-message">No study sessions recorded in the last 7 days</div>';
        return;
    }
    
    // Create chart
    window.dailyStudyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: days,
            datasets: [{
                label: 'Study Hours',
                data: dailyHours,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + 'h';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(1) + ' hours';
                        }
                    }
                }
            }
        }
    });
}

// Create subject performance chart with real data
function createSubjectPerformanceChart(subjectStats) {
    const ctx = document.getElementById('subject-performance-chart').getContext('2d');
    
    // Clear any existing chart
    if (window.subjectPerformanceChart) {
        window.subjectPerformanceChart.destroy();
    }
    
    // Remove any existing empty state message
    const emptyMessage = ctx.canvas.parentNode.querySelector('.empty-chart-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Reset canvas display
    ctx.canvas.style.display = 'block';
    
    // If no data, show empty state
    if (!subjectStats || subjectStats.length === 0) {
        ctx.canvas.style.display = 'none';
        ctx.canvas.parentNode.innerHTML += '<div class="empty-chart-message">No data available for the selected time period</div>';
        return;
    }
    
    // Prepare data
    const labels = subjectStats.map(subject => subject.subject_name);
    const completionRates = subjectStats.map(subject => subject.completion_percentage);
    const colors = subjectStats.map(subject => subject.color || getRandomColor());
    
    // Create chart
    window.subjectPerformanceChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Completion Rate',
                data: completionRates,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: colors,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: colors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Display subject statistics with real data
function displaySubjectStats(subjectStats) {
    const subjectStatsContainer = document.getElementById('subject-stats');
    subjectStatsContainer.innerHTML = '';
    
    // Hide loading spinner
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // If no subjects, show message
    if (!subjectStats || subjectStats.length === 0) {
        subjectStatsContainer.innerHTML = `
            <div class="empty-state">
                <p>No subject statistics available for the selected time period.</p>
            </div>
        `;
        return;
    }
    
    // Sort subjects by study hours
    const sortedSubjects = [...subjectStats].sort((a, b) => b.study_hours - a.study_hours);
    
    // Display subject stats
    sortedSubjects.forEach(subject => {
        const subjectCard = document.createElement('div');
        subjectCard.className = 'subject-stat-card';
        
        subjectCard.innerHTML = `
            <div class="subject-header" style="background-color: ${subject.color || getRandomColor()}">
                <h3>${subject.subject_name}</h3>
            </div>
            <div class="subject-stats-content">
                <div class="stat-row">
                    <div class="stat-item">
                        <span class="stat-label">Study Time</span>
                        <span class="stat-value">${subject.study_hours.toFixed(1)}h</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Assignments</span>
                        <span class="stat-value">${subject.completed_assignments}/${subject.total_assignments}</span>
                    </div>
                </div>
                <div class="subject-progress">
                    <div class="progress-label">
                        <span>Completion Rate</span>
                        <span>${subject.completion_percentage.toFixed(1)}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${subject.completion_percentage}%"></div>
                    </div>
                </div>
            </div>
        `;
        
        subjectStatsContainer.appendChild(subjectCard);
    });
}

// Open export modal
function openExportModal() {
    // Show modal
    document.getElementById('export-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Close all modals
function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    document.body.classList.remove('modal-open');
}

// Handle export with real data
async function handleExport() {
    try {
        // Get export options
        const exportAssignments = document.getElementById('export-assignments').checked;
        const exportEvents = document.getElementById('export-events').checked;
        const exportStudySessions = document.getElementById('export-study-sessions').checked;
        const exportGoals = document.getElementById('export-goals').checked;
        
        const format = document.querySelector('input[name="export-format"]:checked').value;
        
        // Get date range
        const { startDate, endDate } = window.currentDateRange;
        
        // Show loading state
        document.getElementById('export-confirm-btn').textContent = 'Exporting...';
        document.getElementById('export-confirm-btn').disabled = true;
        
        const token = localStorage.getItem('accessToken');
        const exportData = {};
        
        // Fetch data based on selected options
        const fetchPromises = [];
        
        if (exportAssignments) {
            fetchPromises.push(
                fetch(`https://study-o5hp.onrender.com/assignments?due_after=${startDate.toISOString()}&due_before=${endDate.toISOString()}`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }).then(res => res.json()).then(data => { exportData.assignments = data; })
            );
        }
        
        if (exportEvents) {
            fetchPromises.push(
                fetch(`https://study-o5hp.onrender.com/events?start_after=${startDate.toISOString()}&start_before=${endDate.toISOString()}`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }).then(res => res.json()).then(data => { exportData.events = data; })
            );
        }
        
        if (exportStudySessions) {
            fetchPromises.push(
                fetch(`https://study-o5hp.onrender.com/study-sessions?scheduled_after=${startDate.toISOString()}&scheduled_before=${endDate.toISOString()}`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }).then(res => res.json()).then(data => { exportData.study_sessions = data; })
            );
        }
        
        if (exportGoals) {
            fetchPromises.push(
                fetch(`https://study-o5hp.onrender.com/goals`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                }).then(res => res.json()).then(data => { 
                    // Filter goals within date range
                    exportData.goals = data.filter(goal => {
                        const createdAt = new Date(goal.created_at);
                        return createdAt >= startDate && createdAt <= endDate;
                    });
                })
            );
        }
        
        // Wait for all fetch operations to complete
        await Promise.all(fetchPromises);
        
        // Generate export file based on format
        let exportContent, fileName, fileType;
        
        if (format === 'csv') {
            exportContent = generateCSV(exportData);
            fileName = `study_statistics_${formatDateForFile(startDate)}_to_${formatDateForFile(endDate)}.csv`;
            fileType = 'text/csv';
        } else if (format === 'json') {
            exportContent = JSON.stringify(exportData, null, 2);
            fileName = `study_statistics_${formatDateForFile(startDate)}_to_${formatDateForFile(endDate)}.json`;
            fileType = 'application/json';
        } else if (format === 'pdf') {
            // For PDF, we'd normally use a library like jsPDF
            // For this implementation, we'll generate a simple PDF with basic data
            
            // Create a form to submit to a PDF generation service
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = 'https://study-o5hp.onrender.com/export/pdf';
            form.target = '_blank';
            
            // Add the data as a hidden field
            const dataField = document.createElement('input');
            dataField.type = 'hidden';
            dataField.name = 'data';
            dataField.value = JSON.stringify(exportData);
            form.appendChild(dataField);
            
            // Add the token as a hidden field
            const tokenField = document.createElement('input');
            tokenField.type = 'hidden';
            tokenField.name = 'token';
            tokenField.value = token;
            form.appendChild(tokenField);
            
            // Submit the form
            document.body.appendChild(form);
            form.submit();
            document.body.removeChild(form);
            
            // Reset button state
            document.getElementById('export-confirm-btn').textContent = 'Export';
            document.getElementById('export-confirm-btn').disabled = false;
            
            // Close modal
            closeModals();
            return;
        }
        
        // Create download link for CSV or JSON
        const blob = new Blob([exportContent], { type: fileType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 0);
        
        // Reset button state
        document.getElementById('export-confirm-btn').textContent = 'Export';
        document.getElementById('export-confirm-btn').disabled = false;
        
        // Close modal
        closeModals();
        
    } catch (error) {
        console.error('Error exporting data:', error);
        
        // Show error message
        alert('Failed to export data. Please try again later.');
        
        // Reset button state
        document.getElementById('export-confirm-btn').textContent = 'Export';
        document.getElementById('export-confirm-btn').disabled = false;
    }
}

// Generate CSV content from real data
function generateCSV(data) {
    let csvContent = '';
    
    // Add assignments
    if (data.assignments && data.assignments.length > 0) {
        csvContent += 'ASSIGNMENTS\n';
        csvContent += 'ID,Title,Description,Due Date,Priority,Status,Subject ID\n';
        
        data.assignments.forEach(assignment => {
            csvContent += `"${assignment._id}","${assignment.title}","${assignment.description || ''}","${formatDate(new Date(assignment.due_date))}","${assignment.priority}","${assignment.status}","${assignment.subject_id}"\n`;
        });
        
        csvContent += '\n';
    }
    
    // Add events
    if (data.events && data.events.length > 0) {
        csvContent += 'EVENTS\n';
        csvContent += 'ID,Title,Description,Start Time,End Time,Type,Subject ID\n';
        
        data.events.forEach(event => {
            csvContent += `"${event._id}","${event.title}","${event.description || ''}","${formatDate(new Date(event.start_time))}","${formatDate(new Date(event.end_time))}","${event.type}","${event.subject_id || ''}"\n`;
        });
        
        csvContent += '\n';
    }
    
    // Add study sessions
    if (data.study_sessions && data.study_sessions.length > 0) {
        csvContent += 'STUDY SESSIONS\n';
        csvContent += 'ID,Subject ID,Description,Scheduled Date,Planned Duration,Actual Duration,Completed,Completed At\n';
        
        data.study_sessions.forEach(session => {
            csvContent += `"${session._id}","${session.subject_id}","${session.description || ''}","${formatDate(new Date(session.scheduled_date))}","${session.planned_duration} min","${session.actual_duration || 0} min","${session.completed}","${session.completed_at ? formatDate(new Date(session.completed_at)) : ''}"\n`;
        });
        
        csvContent += '\n';
    }
    
    // Add goals
    if (data.goals && data.goals.length > 0) {
        csvContent += 'GOALS\n';
        csvContent += 'ID,Title,Description,Target Date,Subject ID,Progress,Completed,Completed At\n';
        
        data.goals.forEach(goal => {
            csvContent += `"${goal._id}","${goal.title}","${goal.description || ''}","${formatDate(new Date(goal.target_date))}","${goal.subject_id || ''}","${goal.progress}%","${goal.completed}","${goal.completed_at ? formatDate(new Date(goal.completed_at)) : ''}"\n`;
        });
        
        // Add milestones for each goal
        if (data.goals.some(goal => goal.milestones && goal.milestones.length > 0)) {
            csvContent += '\nMILESTONES\n';
            csvContent += 'Goal ID,Title,Completed\n';
            
            data.goals.forEach(goal => {
                if (goal.milestones && goal.milestones.length > 0) {
                    goal.milestones.forEach(milestone => {
                        csvContent += `"${goal._id}","${milestone.title}","${milestone.completed}"\n`;
                    });
                }
            });
        }
    }
    
    return csvContent;
}

// Helper function to get random color for charts
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

// Helper functions for date formatting
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' };
    return date.toLocaleDateString('en-US', options);
}

function formatDayName(date) {
    const options = { weekday: 'short' };
    return date.toLocaleDateString('en-US', options);
}

function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
}

function formatDateForFile(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    return `${year}${month}${day}`;
}

// Logout function
function logout() {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('userData');
    window.location.href = 'login.html';
}

