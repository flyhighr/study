// events.js - For the events page functionality

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
    
    // Initialize calendar
    initCalendar();
    
    // Load events data
    loadEvents();
    
    // Initialize event listeners
    initEventListeners();
    
    // Check URL for new event parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('new') === 'true') {
        openAddEventModal();
    }
    
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

function initCalendar() {
    const calendarEl = document.getElementById('calendar');
    
    if (!calendarEl) return;
    
    window.calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'dayGridMonth',
        headerToolbar: {
            left: 'prev,next today',
            center: 'title',
            right: 'dayGridMonth,timeGridWeek,timeGridDay'
        },
        themeSystem: 'standard',
        selectable: true,
        selectMirror: true,
        navLinks: true,
        editable: false,
        dayMaxEvents: true,
        select: function(arg) {
            // Open add event modal with selected date
            openAddEventModal(arg.start, arg.end);
        },
        eventClick: function(arg) {
            // Open edit event modal
            openEditEventModal(arg.event.id);
        },
        eventTimeFormat: {
            hour: 'numeric',
            minute: '2-digit',
            meridiem: 'short'
        },
        eventDidMount: function(info) {
            // Add tooltip with event details
            const tooltip = document.createElement('div');
            tooltip.className = 'event-tooltip';
            tooltip.innerHTML = `
                <strong>${info.event.title}</strong><br>
                ${info.event.extendedProps.description || ''}<br>
                <em>${formatTime(info.event.start)} - ${formatTime(info.event.end)}</em><br>
                Type: ${capitalizeFirstLetter(info.event.extendedProps.type)}<br>
                Subject: ${info.event.extendedProps.subject}
            `;
            
            // Add hover functionality (this is a simple tooltip, could be enhanced with a library)
            info.el.addEventListener('mouseover', function() {
                document.body.appendChild(tooltip);
                const rect = info.el.getBoundingClientRect();
                tooltip.style.position = 'absolute';
                tooltip.style.top = `${rect.bottom + window.scrollY + 5}px`;
                tooltip.style.left = `${rect.left + window.scrollX}px`;
                tooltip.style.zIndex = 1000;
            });
            
            info.el.addEventListener('mouseout', function() {
                if (document.body.contains(tooltip)) {
                    document.body.removeChild(tooltip);
                }
            });
        }
    });
    
    window.calendar.render();
}

// Initialize event listeners
function initEventListeners() {
    // Add event button
    document.getElementById('add-event-btn').addEventListener('click', () => openAddEventModal());
    document.getElementById('empty-add-btn').addEventListener('click', () => openAddEventModal());
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Event form submission
    document.getElementById('event-form').addEventListener('submit', handleEventFormSubmit);
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteEvent);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });
    
    // Search functionality
    document.getElementById('event-search').addEventListener('input', debounce(filterEvents, 300));
    
    // Filters
    document.getElementById('event-type-filter').addEventListener('change', filterEvents);
    document.getElementById('subject-filter').addEventListener('change', filterEvents);
    document.getElementById('date-range-filter').addEventListener('change', filterEvents);
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModals();
            }
        });
    });
}

async function loadEvents() {
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
        
        // Keep only the first option (All Subjects / No Subject)
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
        
        // Fetch all events
        const eventsResponse = await fetch('https://study-o5hp.onrender.com/events', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!eventsResponse.ok) {
            throw new Error('Failed to fetch events');
        }
        
        const events = await eventsResponse.json();
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store events in a global variable for filtering
        window.allEvents = events;
        
        // Display events
        displayEvents(events, subjectsMap);
        
        // Add events to calendar
        updateCalendarEvents(events, subjectsMap);
        
    } catch (error) {
        console.error('Error loading events:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const eventsList = document.getElementById('events-list');
        eventsList.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load events. Please try again later.</p>
            </div>
        `;
    }
}

function displayEvents(events, subjectsMap) {
    const eventsList = document.getElementById('events-list');
    const noEvents = document.getElementById('no-events');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Check if there are any events
    if (events.length === 0) {
        eventsList.innerHTML = '';
        noEvents.style.display = 'block';
        return;
    }
    
    // Hide no events message
    noEvents.style.display = 'none';
    
    // Clear existing events
    eventsList.innerHTML = '';
    
    // Sort events by start time
    events.sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
    
    // Group events by date
    const eventsByDate = {};
    
    events.forEach(event => {
        const startDate = new Date(event.start_time);
        const dateKey = startDate.toDateString();
        
        if (!eventsByDate[dateKey]) {
            eventsByDate[dateKey] = [];
        }
        
        eventsByDate[dateKey].push(event);
    });
    
    // Add events to list grouped by date
    Object.keys(eventsByDate).forEach(dateKey => {
        const date = new Date(dateKey);
        const dateHeader = document.createElement('div');
        dateHeader.className = 'date-header';
        dateHeader.innerHTML = `
            <h3>${formatDateHeader(date)}</h3>
        `;
        
        eventsList.appendChild(dateHeader);
        
        // Add events for this date
        eventsByDate[dateKey].forEach(event => {
            const subject = event.subject_id ? (subjectsMap[event.subject_id] || { name: 'No Subject', color: '#808080' }) : { name: 'No Subject', color: '#808080' };
            const startTime = new Date(event.start_time);
            const endTime = new Date(event.end_time);
            const isPast = endTime < new Date();
            
            const eventItem = document.createElement('div');
            eventItem.className = `event-item ${isPast ? 'past' : ''}`;
            eventItem.dataset.id = event._id;
            
            eventItem.innerHTML = `
                <div class="event-time">
                    <span class="start-time">${formatTime(startTime)}</span>
                    <span class="time-separator">-</span>
                    <span class="end-time">${formatTime(endTime)}</span>
                </div>
                <div class="event-content">
                    <div class="event-header">
                        <h4>${event.title}</h4>
                        <div class="event-actions">
                            <button class="btn btn-icon edit-btn" data-id="${event._id}">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn btn-icon delete-btn" data-id="${event._id}">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="event-description">
                        ${event.description || 'No description provided'}
                    </div>
                    <div class="event-meta">
                        <span class="event-type ${event.type}">
                            <i class="fas ${getEventTypeIcon(event.type)}"></i>
                            ${capitalizeFirstLetter(event.type)}
                        </span>
                        <span class="subject-tag" style="background-color: ${subject.color}">
                            ${subject.name}
                        </span>
                    </div>
                </div>
            `;
            
            eventsList.appendChild(eventItem);
        });
    });
    
    // Add event listeners to edit and delete buttons
    document.querySelectorAll('.edit-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event bubbling
            const eventId = this.dataset.id;
            openEditEventModal(eventId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event bubbling
            const eventId = this.dataset.id;
            console.log(`Delete button clicked for event: ${eventId}`);
            openDeleteConfirmationModal(eventId);
        });
    });
}

// Update calendar events
function updateCalendarEvents(events, subjectsMap) {
    if (!window.calendar) return;
    
    // Remove all events
    window.calendar.removeAllEvents();
    
    // Add events to calendar
    events.forEach(event => {
        const subject = event.subject_id ? (subjectsMap[event.subject_id] || { name: 'No Subject', color: '#808080' }) : { name: 'No Subject', color: '#808080' };
        
        window.calendar.addEvent({
            id: event._id,
            title: event.title,
            start: event.start_time,
            end: event.end_time,
            backgroundColor: getEventColor(event.type, subject.color),
            borderColor: getEventColor(event.type, subject.color),
            textColor: '#ffffff',
            extendedProps: {
                description: event.description,
                type: event.type,
                subject: subject.name
            }
        });
    });
}

// Filter events
function filterEvents() {
    const searchTerm = document.getElementById('event-search').value.toLowerCase();
    const typeFilter = document.getElementById('event-type-filter').value;
    const subjectFilter = document.getElementById('subject-filter').value;
    const dateRangeFilter = document.getElementById('date-range-filter').value;
    
    // Get all events
    const events = window.allEvents || [];
    
    // Apply filters
    const filteredEvents = events.filter(event => {
        // Search term filter
        const matchesSearch = 
            event.title.toLowerCase().includes(searchTerm) ||
            (event.description && event.description.toLowerCase().includes(searchTerm));
        
        // Type filter
        const matchesType = typeFilter === 'all' || event.type === typeFilter;
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || 
            (event.subject_id && event.subject_id === subjectFilter) || 
            (subjectFilter === 'none' && !event.subject_id);
        
        // Date range filter
        let matchesDateRange = true;
        const eventDate = new Date(event.start_time);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        if (dateRangeFilter === 'today') {
            const tomorrow = new Date(today);
            tomorrow.setDate(tomorrow.getDate() + 1);
            matchesDateRange = eventDate >= today && eventDate < tomorrow;
        } else if (dateRangeFilter === 'this_week') {
            const nextWeek = new Date(today);
            nextWeek.setDate(nextWeek.getDate() + 7);
            matchesDateRange = eventDate >= today && eventDate < nextWeek;
        } else if (dateRangeFilter === 'next_week') {
            const nextWeekStart = new Date(today);
            nextWeekStart.setDate(nextWeekStart.getDate() + 7);
            const nextWeekEnd = new Date(nextWeekStart);
            nextWeekEnd.setDate(nextWeekEnd.getDate() + 7);
            matchesDateRange = eventDate >= nextWeekStart && eventDate < nextWeekEnd;
        } else if (dateRangeFilter === 'this_month') {
            const nextMonth = new Date(today);
            nextMonth.setMonth(nextMonth.getMonth() + 1);
            matchesDateRange = eventDate >= today && eventDate < nextMonth;
        }
        
        return matchesSearch && matchesType && matchesSubject && matchesDateRange;
    });
    
    // Fetch subjects for display
    const token = localStorage.getItem('accessToken');
    fetch('https://study-o5hp.onrender.com/subjects', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(subjects => {
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Display filtered events
        displayEvents(filteredEvents, subjectsMap);
        
        // Update calendar
        updateCalendarEvents(filteredEvents, subjectsMap);
    })
    .catch(error => {
        console.error('Error fetching subjects for filtering:', error);
    });
}

// Open add event modal
function openAddEventModal(start, end) {
    // Reset form
    document.getElementById('event-form').reset();
    document.getElementById('event-id').value = '';
    
    // Set default times if provided
    if (start) {
        document.getElementById('start-time').value = formatDateTimeForInput(start);
        
        // If end is not provided, set it to 1 hour after start
        if (!end) {
            const endTime = new Date(start);
            endTime.setHours(endTime.getHours() + 1);
            end = endTime;
        }
        
        document.getElementById('end-time').value = formatDateTimeForInput(end);
    } else {
        // Set default times to now and 1 hour from now
        const now = new Date();
        const oneHourLater = new Date(now);
        oneHourLater.setHours(oneHourLater.getHours() + 1);
        
        document.getElementById('start-time').value = formatDateTimeForInput(now);
        document.getElementById('end-time').value = formatDateTimeForInput(oneHourLater);
    }
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Add New Event';
    document.getElementById('save-btn').textContent = 'Save Event';
    
    // Show modal
    document.getElementById('event-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit event modal
function openEditEventModal(eventId) {
    const token = localStorage.getItem('accessToken');
    
    // Fetch event details
    fetch(`https://study-o5hp.onrender.com/events/${eventId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(event => {
        // Populate form
        document.getElementById('event-id').value = event._id;
        document.getElementById('title').value = event.title;
        document.getElementById('description').value = event.description || '';
        document.getElementById('start-time').value = formatDateTimeForInput(new Date(event.start_time));
        document.getElementById('end-time').value = formatDateTimeForInput(new Date(event.end_time));
        document.getElementById('type').value = event.type;
        document.getElementById('subject-id').value = event.subject_id || '';
        
        // Update modal title
        document.getElementById('modal-title').textContent = 'Edit Event';
        document.getElementById('save-btn').textContent = 'Update Event';
        
        // Show modal
        document.getElementById('event-modal').classList.add('active');
        document.body.classList.add('modal-open');
    })
    .catch(error => {
        console.error('Error fetching event details:', error);
        alert('Failed to load event details. Please try again.');
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(eventId) {
    // Store event ID for deletion
    document.getElementById('confirm-delete-btn').dataset.id = eventId;
    
    // Show modal
    document.getElementById('delete-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Close all modals
function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    document.body.classList.remove('modal-open');
}

async function handleEventFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const eventId = document.getElementById('event-id').value;
    const isEdit = eventId !== '';
    
    // Get form data
    const formData = {
        title: document.getElementById('title').value,
        description: document.getElementById('description').value,
        start_time: new Date(document.getElementById('start-time').value).toISOString(),
        end_time: new Date(document.getElementById('end-time').value).toISOString(),
        type: document.getElementById('type').value
    };
    
    // Add subject_id if selected
    const subjectId = document.getElementById('subject-id').value;
    if (subjectId) {
        formData.subject_id = subjectId;
    }
    
    // Show loading state
    const saveBtn = document.getElementById('save-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        // Validate that end time is after start time
        if (new Date(formData.end_time) <= new Date(formData.start_time)) {
            throw new Error('End time must be after start time');
        }
        
        let response;
        
        if (isEdit) {
            // Update existing event
            response = await fetch(`https://study-o5hp.onrender.com/events/${eventId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new event
            response = await fetch('https://study-o5hp.onrender.com/events', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        }
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to save event');
        }
        
        const savedEvent = await response.json();
        
        // Update the local data
        if (isEdit) {
            // Update in the allEvents array
            if (window.allEvents) {
                window.allEvents = window.allEvents.map(e => 
                    e._id === eventId ? savedEvent : e
                );
            }
        } else {
            // Add to the allEvents array
            if (window.allEvents) {
                window.allEvents.push(savedEvent);
            } else {
                window.allEvents = [savedEvent];
            }
        }
        
        // Close modal
        closeModals();
        
        // Update the UI with current data
        const subjectsResponse = await fetch('https://study-o5hp.onrender.com/subjects', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (subjectsResponse.ok) {
            const subjects = await subjectsResponse.json();
            const subjectsMap = subjects.reduce((acc, subject) => {
                acc[subject._id] = subject;
                return acc;
            }, {});
            
            // Refresh the displays
            displayEvents(window.allEvents, subjectsMap);
            updateCalendarEvents(window.allEvents, subjectsMap);
        } else {
            // Fallback to full reload
            loadEvents();
        }
        
    } catch (error) {
        console.error('Error saving event:', error);
        alert(error.message);
    } finally {
        // Always reset button state
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

async function deleteEvent() {
    const token = localStorage.getItem('accessToken');
    const eventId = document.getElementById('confirm-delete-btn').dataset.id;
    
    if (!eventId) {
        console.error('No event ID found for deletion');
        alert('Error: Could not identify which event to delete.');
        closeModals();
        return;
    }
    
    try {
        // Show loading state
        const deleteBtn = document.getElementById('confirm-delete-btn');
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
        
        const response = await fetch(`https://study-o5hp.onrender.com/events/${eventId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete event');
        }
        
        // Update the local data
        if (window.allEvents) {
            window.allEvents = window.allEvents.filter(e => e._id !== eventId);
        }
        
        // Remove event from calendar
        if (window.calendar) {
            const calendarEvent = window.calendar.getEventById(eventId);
            if (calendarEvent) {
                calendarEvent.remove();
            }
        }
        
        // Close modal
        closeModals();
        
        // Update the UI with current data
        const subjectsResponse = await fetch('https://study-o5hp.onrender.com/subjects', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (subjectsResponse.ok) {
            const subjects = await subjectsResponse.json();
            const subjectsMap = subjects.reduce((acc, subject) => {
                acc[subject._id] = subject;
                return acc;
            }, {});
            
            // Refresh the display
            displayEvents(window.allEvents, subjectsMap);
        } else {
            // Fallback to full reload
            loadEvents();
        }
        
    } catch (error) {
        console.error('Error deleting event:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        const deleteBtn = document.getElementById('confirm-delete-btn');
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete';
        
        // Clear the event ID
        deleteBtn.dataset.id = '';
    }
}

// Toggle between calendar and list view
function toggleView() {
    const viewType = this.dataset.view;
    
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    this.classList.add('active');
    
    // Show selected view
    if (viewType === 'calendar') {
        document.getElementById('calendar-container').classList.add('view-active');
        document.getElementById('events-list-container').classList.remove('view-active');
        
        // Refresh calendar
        if (window.calendar) {
            window.calendar.updateSize();
        }
    } else {
        document.getElementById('calendar-container').classList.remove('view-active');
        document.getElementById('events-list-container').classList.add('view-active');
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatDateHeader(date) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    if (date.getTime() === today.getTime()) {
        return 'Today, ' + formatDate(date);
    } else if (date.getTime() === tomorrow.getTime()) {
        return 'Tomorrow, ' + formatDate(date);
    } else {
        const options = { weekday: 'long', month: 'short', day: 'numeric', year: 'numeric' };
        return date.toLocaleDateString('en-US', options);
    }
}

function formatTime(date) {
    const options = { hour: 'numeric', minute: '2-digit', hour12: true };
    return date.toLocaleTimeString('en-US', options);
}

function formatDateTimeForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function getEventTypeIcon(type) {
    switch (type) {
        case 'exam':
            return 'fa-book';
        case 'holiday':
            return 'fa-umbrella-beach';
        case 'personal':
        default:
            return 'fa-user';
    }
}

function getEventColor(type, subjectColor) {
    switch (type) {
        case 'exam':
            return '#e74c3c'; // Red
        case 'holiday':
            return '#3498db'; // Blue
        case 'personal':
        default:
            return subjectColor || '#2ecc71'; // Green or subject color
    }
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