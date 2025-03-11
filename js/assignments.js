// assignments.js - For the assignments page functionality

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
    
    // Load assignments data
    loadAssignments();
    
    // Initialize event listeners
    initEventListeners();
    
    // Check URL for new assignment parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('new') === 'true') {
        openAddAssignmentModal();
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

// Initialize event listeners
function initEventListeners() {
    // Add assignment button
    document.getElementById('add-assignment-btn').addEventListener('click', openAddAssignmentModal);
    document.getElementById('empty-add-btn').addEventListener('click', openAddAssignmentModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Assignment form submission
    document.getElementById('assignment-form').addEventListener('submit', handleAssignmentFormSubmit);
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteAssignment);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });

    // Share assignments button
    document.getElementById('share-assignments-btn').addEventListener('click', createShareableLink);
    
    // Copy link button
    document.getElementById('copy-link-btn').addEventListener('click', copyShareLink);
    
    // Share close button
    document.getElementById('share-close-btn').addEventListener('click', closeModals);
    
    // Search functionality
    document.getElementById('assignment-search').addEventListener('input', debounce(filterAssignments, 300));
    
    // Filters
    document.getElementById('status-filter').addEventListener('change', filterAssignments);
    document.getElementById('subject-filter').addEventListener('change', filterAssignments);
    document.getElementById('priority-filter').addEventListener('change', filterAssignments);
    document.getElementById('due-date-filter').addEventListener('change', filterAssignments);
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModals();
            }
        });
    });
}

// Create shareable link function
async function createShareableLink() {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading state
        const shareBtn = document.getElementById('share-assignments-btn');
        shareBtn.disabled = true;
        shareBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating link...';
        
        const response = await fetch('https://study-o5hp.onrender.com/assignments/share', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to create shareable link');
        }
        
        const data = await response.json();
        
        // Reset share button
        shareBtn.disabled = false;
        shareBtn.innerHTML = '<i class="fas fa-share-alt"></i> Share';
        
        // Show the share modal with the link
        document.getElementById('share-link').value = data.share_link;
        document.getElementById('share-modal').classList.add('active');
        document.body.classList.add('modal-open');
        
        // Select the link text for easy copying
        document.getElementById('share-link').select();
        
    } catch (error) {
        console.error('Error creating shareable link:', error);
        
        // Reset share button
        const shareBtn = document.getElementById('share-assignments-btn');
        shareBtn.disabled = false;
        shareBtn.innerHTML = '<i class="fas fa-share-alt"></i> Share';
        
        // Show error message
        alert('Failed to create shareable link. Please try again.');
    }
}

// Copy share link function
function copyShareLink() {
    const shareLink = document.getElementById('share-link');
    shareLink.select();
    document.execCommand('copy');
    
    // Change button text temporarily
    const copyBtn = document.getElementById('copy-link-btn');
    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    
    setTimeout(() => {
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
    }, 2000);
}

// Fix for the loadAssignments function to prevent duplicate subjects
async function loadAssignments() {
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
        
        // Keep only the first option (All Subjects / Select a subject)
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
        
        // Fetch all assignments
        const assignmentsResponse = await fetch('https://study-o5hp.onrender.com/assignments', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!assignmentsResponse.ok) {
            throw new Error('Failed to fetch assignments');
        }
        
        const assignments = await assignmentsResponse.json();
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store assignments in a global variable for filtering
        window.allAssignments = assignments;
        
        // Display assignments
        displayAssignments(assignments, subjectsMap);
        
    } catch (error) {
        console.error('Error loading assignments:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const assignmentsList = document.getElementById('assignments-list');
        assignmentsList.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load assignments. Please try again later.</p>
            </div>
        `;
    }
}

function displayAssignments(assignments, subjectsMap) {
    const assignmentsList = document.getElementById('assignments-list');
    const assignmentsGrid = document.getElementById('assignments-grid');
    const noAssignments = document.getElementById('no-assignments');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Update counts
    document.getElementById('total-assignments-count').textContent = window.allAssignments.length;
    document.getElementById('shown-assignments-count').textContent = assignments.length;
    
    // Check if there are any assignments
    if (assignments.length === 0) {
        assignmentsList.innerHTML = '';
        assignmentsGrid.innerHTML = '';
        noAssignments.style.display = 'block';
        return;
    }
    
    // Hide no assignments message
    noAssignments.style.display = 'none';
    
    // Clear existing assignments
    assignmentsList.innerHTML = '';
    assignmentsGrid.innerHTML = '';
    
    // Add assignments to list and grid views
    assignments.forEach(assignment => {
        const subject = subjectsMap[assignment.subject_id] || { name: 'No Subject', color: '#808080' };
        const dueDate = new Date(assignment.due_date);
        const isOverdue = dueDate < new Date() && assignment.status !== 'completed';
        
        // Create list item
        const listItem = document.createElement('div');
        listItem.className = 'assignment-item';
        listItem.dataset.id = assignment._id;
        
        listItem.innerHTML = `
            <div class="assignment-status ${assignment.status}"></div>
            <div class="assignment-content">
                <div class="assignment-header">
                    <h3>${assignment.title}</h3>
                    <div class="assignment-actions">
                        <button class="btn btn-icon edit-btn" data-id="${assignment._id}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon delete-btn" data-id="${assignment._id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="assignment-description">
                    ${assignment.description || 'No description provided'}
                </div>
                <div class="assignment-meta">
                    <span class="subject-tag" style="background-color: ${subject.color}">
                        ${subject.name}
                    </span>
                    <span class="due-date ${isOverdue ? 'overdue' : ''}">
                        <i class="fas fa-calendar-alt"></i>
                        ${formatDate(dueDate)}
                    </span>
                    <span class="priority ${assignment.priority}">
                        <i class="fas fa-flag"></i>
                        ${capitalizeFirstLetter(assignment.priority)}
                    </span>
                </div>
            </div>
        `;
        
        // Create grid item
        const gridItem = document.createElement('div');
        gridItem.className = 'assignment-card';
        gridItem.dataset.id = assignment._id;
        
        gridItem.innerHTML = `
            <div class="card-header" style="background-color: ${subject.color}">
                <span class="status-badge ${assignment.status}">${capitalizeFirstLetter(assignment.status)}</span>
                <div class="card-actions">
                    <button class="btn btn-icon edit-btn" data-id="${assignment._id}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${assignment._id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <h3>${assignment.title}</h3>
                <p class="card-description">${assignment.description || 'No description provided'}</p>
                <div class="card-meta">
                    <div class="meta-item">
                        <i class="fas fa-calendar-alt"></i>
                        <span class="${isOverdue ? 'overdue' : ''}">${formatDate(dueDate)}</span>
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-book"></i>
                        <span>${subject.name}</span>
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-flag"></i>
                        <span class="${assignment.priority}">${capitalizeFirstLetter(assignment.priority)}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Add to containers
        assignmentsList.appendChild(listItem);
        assignmentsGrid.appendChild(gridItem);
    });
    
    // Remove any existing event listeners (to prevent duplicates)
    document.querySelectorAll('.edit-btn, .delete-btn').forEach(btn => {
        btn.replaceWith(btn.cloneNode(true));
    });
    
    // Add event listeners to edit buttons
    document.querySelectorAll('.edit-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event bubbling
            const assignmentId = this.dataset.id;
            openEditAssignmentModal(assignmentId);
        });
    });
    
    // Add event listeners to delete buttons
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event bubbling
            const assignmentId = this.dataset.id;
            console.log(`Delete button clicked for assignment: ${assignmentId}`);
            openDeleteConfirmationModal(assignmentId);
        });
    });
}

// Filter assignments
function filterAssignments() {
    const searchTerm = document.getElementById('assignment-search').value.toLowerCase();
    const statusFilter = document.getElementById('status-filter').value;
    const subjectFilter = document.getElementById('subject-filter').value;
    const priorityFilter = document.getElementById('priority-filter').value;
    const dueDateFilter = document.getElementById('due-date-filter').value;
    
    // Get all assignments
    const assignments = window.allAssignments || [];
    
    // Apply filters
    const filteredAssignments = assignments.filter(assignment => {
        // Search term filter
        const matchesSearch = 
            assignment.title.toLowerCase().includes(searchTerm) ||
            (assignment.description && assignment.description.toLowerCase().includes(searchTerm));
        
        // Status filter
        const matchesStatus = statusFilter === 'all' || assignment.status === statusFilter;
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || assignment.subject_id === subjectFilter;
        
        // Priority filter
        const matchesPriority = priorityFilter === 'all' || assignment.priority === priorityFilter;
        
        // Due date filter
        let matchesDueDate = true;
        const dueDate = new Date(assignment.due_date);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        if (dueDateFilter === 'today') {
            const tomorrow = new Date(today);
            tomorrow.setDate(tomorrow.getDate() + 1);
            matchesDueDate = dueDate >= today && dueDate < tomorrow;
        } else if (dueDateFilter === 'this_week') {
            const nextWeek = new Date(today);
            nextWeek.setDate(nextWeek.getDate() + 7);
            matchesDueDate = dueDate >= today && dueDate < nextWeek;
        } else if (dueDateFilter === 'next_week') {
            const nextWeekStart = new Date(today);
            nextWeekStart.setDate(nextWeekStart.getDate() + 7);
            const nextWeekEnd = new Date(nextWeekStart);
            nextWeekEnd.setDate(nextWeekEnd.getDate() + 7);
            matchesDueDate = dueDate >= nextWeekStart && dueDate < nextWeekEnd;
        } else if (dueDateFilter === 'this_month') {
            const nextMonth = new Date(today);
            nextMonth.setMonth(nextMonth.getMonth() + 1);
            matchesDueDate = dueDate >= today && dueDate < nextMonth;
        }
        
        return matchesSearch && matchesStatus && matchesSubject && matchesPriority && matchesDueDate;
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
        
        // Display filtered assignments
        displayAssignments(filteredAssignments, subjectsMap);
    })
    .catch(error => {
        console.error('Error fetching subjects for filtering:', error);
    });
}

// Open add assignment modal
function openAddAssignmentModal() {
    // Reset form
    document.getElementById('assignment-form').reset();
    document.getElementById('assignment-id').value = '';
    
    // Set default due date to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(23, 59);
    document.getElementById('due-date').value = formatDateTimeForInput(tomorrow);
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Add New Assignment';
    document.getElementById('save-btn').textContent = 'Save Assignment';
    
    // Show modal
    document.getElementById('assignment-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit assignment modal
function openEditAssignmentModal(assignmentId) {
    const token = localStorage.getItem('accessToken');
    
    // Fetch assignment details
    fetch(`https://study-o5hp.onrender.com/assignments/${assignmentId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(assignment => {
        // Populate form
        document.getElementById('assignment-id').value = assignment._id;
        document.getElementById('title').value = assignment.title;
        document.getElementById('description').value = assignment.description || '';
        document.getElementById('due-date').value = formatDateTimeForInput(new Date(assignment.due_date));
        document.getElementById('subject-id').value = assignment.subject_id;
        document.getElementById('priority').value = assignment.priority;
        document.getElementById('status').value = assignment.status;
        
        // Update modal title
        document.getElementById('modal-title').textContent = 'Edit Assignment';
        document.getElementById('save-btn').textContent = 'Update Assignment';
        
        // Show modal
        document.getElementById('assignment-modal').classList.add('active');
        document.body.classList.add('modal-open');
    })
    .catch(error => {
        console.error('Error fetching assignment details:', error);
        alert('Failed to load assignment details. Please try again.');
    });
}

function openDeleteConfirmationModal(assignmentId) {
    if (!assignmentId) {
        console.error('No assignment ID provided for deletion');
        return;
    }
    
    console.log(`Opening delete modal for assignment: ${assignmentId}`);
    
    // Store assignment ID for deletion
    const deleteBtn = document.getElementById('confirm-delete-btn');
    deleteBtn.dataset.id = assignmentId;
    
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

// Fix for the handleAssignmentFormSubmit function
async function handleAssignmentFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const assignmentId = document.getElementById('assignment-id').value;
    const isEdit = assignmentId !== '';
    
    // Get form data
    const formData = {
        title: document.getElementById('title').value,
        description: document.getElementById('description').value,
        due_date: new Date(document.getElementById('due-date').value).toISOString(),
        subject_id: document.getElementById('subject-id').value,
        priority: document.getElementById('priority').value,
        status: document.getElementById('status').value
    };
    
    // Show loading state
    const saveBtn = document.getElementById('save-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        let response;
        
        if (isEdit) {
            // Update existing assignment
            response = await fetch(`https://study-o5hp.onrender.com/assignments/${assignmentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new assignment
            response = await fetch('https://study-o5hp.onrender.com/assignments', {
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
            throw new Error(errorData.detail || 'Failed to save assignment');
        }
        
        const savedAssignment = await response.json();
        
        // Update the local data
        if (isEdit) {
            // Update in the allAssignments array
            if (window.allAssignments) {
                window.allAssignments = window.allAssignments.map(a => 
                    a._id === assignmentId ? savedAssignment : a
                );
            }
        } else {
            // Add to the allAssignments array
            if (window.allAssignments) {
                window.allAssignments.push(savedAssignment);
            } else {
                window.allAssignments = [savedAssignment];
            }
        }
        
        // Close modal and reload assignments
        closeModals();
        
        // Refresh the display without full reload
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
            
            displayAssignments(window.allAssignments, subjectsMap);
        } else {
            // Fallback to full reload if subjects fetch fails
            loadAssignments();
        }
        
    } catch (error) {
        console.error('Error saving assignment:', error);
        alert(error.message);
    } finally {
        // Always reset button state
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

async function deleteAssignment() {
    const token = localStorage.getItem('accessToken');
    const assignmentId = document.getElementById('confirm-delete-btn').dataset.id;
    
    if (!assignmentId) {
        console.error('No assignment ID found for deletion');
        alert('Error: Could not identify which assignment to delete.');
        closeModals();
        return;
    }
    
    try {
        // Show loading state
        const deleteBtn = document.getElementById('confirm-delete-btn');
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
        
        const response = await fetch(`https://study-o5hp.onrender.com/assignments/${assignmentId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete assignment');
        }
        
        // Update the local data
        if (window.allAssignments) {
            window.allAssignments = window.allAssignments.filter(a => a._id !== assignmentId);
        }
        
        // Close modal
        closeModals();
        
        // Completely refresh the assignments display
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
            
            // Completely redraw the assignments
            displayAssignments(window.allAssignments, subjectsMap);
        } else {
            // Full reload as fallback
            loadAssignments();
        }
        
    } catch (error) {
        console.error('Error deleting assignment:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Reset delete button
        const deleteBtn = document.getElementById('confirm-delete-btn');
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete';
        
        // Clear the assignment ID
        deleteBtn.dataset.id = '';
    }
}

// Toggle between list and grid view
function toggleView() {
    const viewType = this.dataset.view;
    
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    this.classList.add('active');
    
    // Show selected view
    if (viewType === 'list') {
        document.getElementById('assignments-list').classList.add('view-active');
        document.getElementById('assignments-grid').classList.remove('view-active');
    } else {
        document.getElementById('assignments-list').classList.remove('view-active');
        document.getElementById('assignments-grid').classList.add('view-active');
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatDateTimeForInput(date) {
    return date.toISOString().slice(0, 16);
}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
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