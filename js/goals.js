// goals.js - For the goals page functionality

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
    
    // Load goals data
    loadGoals();
    
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

// Initialize event listeners
function initEventListeners() {
    // Add goal button
    document.getElementById('add-goal-btn').addEventListener('click', openAddGoalModal);
    document.getElementById('empty-add-btn').addEventListener('click', openAddGoalModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Goal form submission
    document.getElementById('goal-form').addEventListener('submit', handleGoalFormSubmit);
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteGoal);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });
    
    // Search functionality
    document.getElementById('goal-search').addEventListener('input', debounce(filterGoals, 300));
    
    // Filters
    document.getElementById('status-filter').addEventListener('change', filterGoals);
    document.getElementById('subject-filter').addEventListener('change', filterGoals);
    document.getElementById('time-filter').addEventListener('change', filterGoals);
    
    // Add milestone button
    document.getElementById('add-milestone-btn').addEventListener('click', addMilestone);
}

// Load goals data
async function loadGoals() {
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
        
        // Clear existing options to prevent duplicates
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
        
        // Fetch all goals
        const goalsResponse = await fetch('https://study-o5hp.onrender.com/goals', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!goalsResponse.ok) {
            throw new Error('Failed to fetch goals');
        }
        
        const goals = await goalsResponse.json();
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store goals and subjects in global variables for filtering
        window.allGoals = goals;
        window.subjectsMap = subjectsMap;
        
        // Update overview statistics
        updateGoalStats(goals);
        
        // Display goals
        displayGoals(goals, subjectsMap);
        
    } catch (error) {
        console.error('Error loading goals:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const inProgressColumn = document.getElementById('in-progress-column');
        inProgressColumn.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load goals. Please try again later.</p>
                <p class="error-details">${error.message}</p>
            </div>
        `;
    }
}

// Update goal statistics
function updateGoalStats(goals) {
    const totalGoals = goals.length;
    const completedGoals = goals.filter(goal => goal.completed).length;
    const inProgressGoals = totalGoals - completedGoals;
    const completionRate = totalGoals > 0 ? Math.round((completedGoals / totalGoals) * 100) : 0;
    
    document.getElementById('total-goals').textContent = totalGoals;
    document.getElementById('completed-goals').textContent = completedGoals;
    document.getElementById('in-progress-goals').textContent = inProgressGoals;
    document.getElementById('completion-rate').textContent = `${completionRate}%`;
    
    document.getElementById('in-progress-count').textContent = inProgressGoals;
    document.getElementById('completed-count').textContent = completedGoals;
}

// Display goals
function displayGoals(goals, subjectsMap) {
    const inProgressColumn = document.getElementById('in-progress-column');
    const completedColumn = document.getElementById('completed-column');
    const goalsList = document.getElementById('goals-list');
    const noGoals = document.getElementById('no-goals');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Check if there are any goals
    if (goals.length === 0) {
        inProgressColumn.innerHTML = '';
        completedColumn.innerHTML = '';
        goalsList.innerHTML = '';
        noGoals.style.display = 'block';
        return;
    }
    
    // Hide no goals message
    noGoals.style.display = 'none';
    
    // Clear existing goals
    inProgressColumn.innerHTML = '';
    completedColumn.innerHTML = '';
    goalsList.innerHTML = '';
    
    // Separate goals by status
    const inProgressGoals = goals.filter(goal => !goal.completed);
    const completedGoals = goals.filter(goal => goal.completed);
    
    // Sort goals by target date
    inProgressGoals.sort((a, b) => new Date(a.target_date) - new Date(b.target_date));
    completedGoals.sort((a, b) => new Date(b.completed_at || b.updated_at || b.created_at) - new Date(a.completed_at || a.updated_at || a.created_at));
    
    // Display in-progress goals
    inProgressGoals.forEach(goal => {
        const subject = goal.subject_id ? (subjectsMap[goal.subject_id] || { name: 'No Subject', color: '#808080' }) : { name: 'No Subject', color: '#808080' };
        const targetDate = new Date(goal.target_date);
        const isNearDeadline = isDateNearing(targetDate);
        
        // Create kanban card
        const kanbanCard = createGoalCard(goal, subject, targetDate, isNearDeadline);
        inProgressColumn.appendChild(kanbanCard);
        
        // Create list item
        const listItem = createGoalListItem(goal, subject, targetDate, isNearDeadline);
        goalsList.appendChild(listItem);
    });
    
    // Display completed goals
    completedGoals.forEach(goal => {
        const subject = goal.subject_id ? (subjectsMap[goal.subject_id] || { name: 'No Subject', color: '#808080' }) : { name: 'No Subject', color: '#808080' };
        const targetDate = new Date(goal.target_date);
        const completedDate = new Date(goal.completed_at || goal.updated_at || goal.created_at);
        
        // Create kanban card
        const kanbanCard = createGoalCard(goal, subject, targetDate, false, completedDate);
        completedColumn.appendChild(kanbanCard);
        
        // Create list item
        const listItem = createGoalListItem(goal, subject, targetDate, false, completedDate);
        goalsList.appendChild(listItem);
    });
    
    // Add event listeners to edit and delete buttons
    document.querySelectorAll('.edit-btn').forEach(button => {
        button.addEventListener('click', function() {
            const goalId = this.dataset.id;
            openEditGoalModal(goalId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', function() {
            const goalId = this.dataset.id;
            openDeleteConfirmationModal(goalId);
        });
    });
    
    // Add event listeners to milestone checkboxes
    document.querySelectorAll('.milestone-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const goalId = this.dataset.goalId;
            const milestoneIndex = parseInt(this.dataset.index);
            updateMilestone(goalId, milestoneIndex, this.checked);
        });
    });
    
    // Add event listeners to mark complete buttons
    document.querySelectorAll('.mark-complete-btn').forEach(button => {
        button.addEventListener('click', function() {
            const goalId = this.dataset.id;
            markGoalComplete(goalId);
        });
    });
}

// Create goal card for kanban view
function createGoalCard(goal, subject, targetDate, isNearDeadline, completedDate) {
    const card = document.createElement('div');
    card.className = `goal-card ${isNearDeadline && !goal.completed ? 'near-deadline' : ''}`;
    card.dataset.id = goal._id;
    
    // Calculate progress percentage
    const totalMilestones = goal.milestones ? goal.milestones.length : 0;
    const completedMilestones = goal.milestones ? goal.milestones.filter(m => m.completed).length : 0;
    const progressPercentage = totalMilestones > 0 ? Math.round((completedMilestones / totalMilestones) * 100) : goal.completed ? 100 : 0;
    
    let cardContent = `
        <div class="card-header" style="background-color: ${subject.color}">
            <h3>${goal.title}</h3>
            <div class="card-actions">
                <button class="btn btn-icon edit-btn" data-id="${goal._id}">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-icon delete-btn" data-id="${goal._id}">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
        <div class="card-body">
            <div class="goal-description">
                ${goal.description || 'No description provided'}
            </div>
            <div class="goal-meta">
                <div class="meta-item">
                    <i class="fas fa-calendar-alt"></i>
                    <span>${goal.completed ? 'Completed on ' + formatDate(completedDate) : 'Due by ' + formatDate(targetDate)}</span>
                </div>
                <div class="meta-item">
                    <i class="fas fa-book"></i>
                    <span>${subject.name}</span>
                </div>
            </div>
            <div class="goal-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progressPercentage}%"></div>
                </div>
                <span class="progress-text">${progressPercentage}% Complete</span>
            </div>
    `;
    
    // Add milestones if any
    if (goal.milestones && goal.milestones.length > 0) {
        cardContent += `<div class="goal-milestones">`;
        goal.milestones.forEach((milestone, index) => {
            cardContent += `
                <div class="milestone-item">
                    <input type="checkbox" class="milestone-checkbox" 
                        id="milestone-${goal._id}-${index}" 
                        data-goal-id="${goal._id}" 
                        data-index="${index}" 
                        ${milestone.completed ? 'checked' : ''}
                        ${goal.completed ? 'disabled' : ''}>
                    <label for="milestone-${goal._id}-${index}">${milestone.title}</label>
                </div>
            `;
        });
        cardContent += `</div>`;
    }
    
    // Add mark complete button for in-progress goals
    if (!goal.completed) {
        cardContent += `
            <button class="btn btn-primary mark-complete-btn" data-id="${goal._id}">
                <i class="fas fa-check"></i> Mark Complete
            </button>
        `;
    }
    
    cardContent += `</div>`;
    
    card.innerHTML = cardContent;
    return card;
}

// Create goal list item for list view
function createGoalListItem(goal, subject, targetDate, isNearDeadline, completedDate) {
    const listItem = document.createElement('div');
    listItem.className = `goal-item ${isNearDeadline && !goal.completed ? 'near-deadline' : ''}`;
    listItem.dataset.id = goal._id;
    
    // Calculate progress percentage
    const totalMilestones = goal.milestones ? goal.milestones.length : 0;
    const completedMilestones = goal.milestones ? goal.milestones.filter(m => m.completed).length : 0;
    const progressPercentage = totalMilestones > 0 ? Math.round((completedMilestones / totalMilestones) * 100) : goal.completed ? 100 : 0;
    
    let itemContent = `
        <div class="goal-status ${goal.completed ? 'completed' : 'in-progress'}">
            <i class="fas ${goal.completed ? 'fa-check-circle' : 'fa-spinner'}"></i>
        </div>
        <div class="goal-content">
            <div class="goal-header">
                <h3>${goal.title}</h3>
                <div class="goal-actions">
                    <button class="btn btn-icon edit-btn" data-id="${goal._id}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${goal._id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="goal-description">
                ${goal.description || 'No description provided'}
            </div>
            <div class="goal-meta">
                <span class="subject-tag" style="background-color: ${subject.color}">
                    ${subject.name}
                </span>
                <span class="goal-date">
                    <i class="fas fa-calendar-alt"></i>
                    ${goal.completed ? 'Completed on ' + formatDate(completedDate) : 'Due by ' + formatDate(targetDate)}
                </span>
                <span class="goal-progress-text">
                    <i class="fas fa-tasks"></i>
                    ${progressPercentage}% Complete
                </span>
            </div>
            <div class="goal-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progressPercentage}%"></div>
                </div>
            </div>
    `;
    
    // Add milestones if any
    if (goal.milestones && goal.milestones.length > 0) {
        itemContent += `<div class="goal-milestones">`;
        goal.milestones.forEach((milestone, index) => {
            itemContent += `
                <div class="milestone-item">
                    <input type="checkbox" class="milestone-checkbox" 
                        id="milestone-list-${goal._id}-${index}" 
                        data-goal-id="${goal._id}" 
                        data-index="${index}" 
                        ${milestone.completed ? 'checked' : ''}
                        ${goal.completed ? 'disabled' : ''}>
                    <label for="milestone-list-${goal._id}-${index}">${milestone.title}</label>
                </div>
            `;
        });
        itemContent += `</div>`;
    }
    
    // Add mark complete button for in-progress goals
    if (!goal.completed) {
        itemContent += `
            <div class="goal-actions-footer">
                <button class="btn btn-primary mark-complete-btn" data-id="${goal._id}">
                    <i class="fas fa-check"></i> Mark Complete
                </button>
            </div>
        `;
    }
    
    itemContent += `</div>`;
    
    listItem.innerHTML = itemContent;
    return listItem;
}

// Filter goals
function filterGoals() {
    const searchTerm = document.getElementById('goal-search').value.toLowerCase();
    const statusFilter = document.getElementById('status-filter').value;
    const subjectFilter = document.getElementById('subject-filter').value;
    const timeFilter = document.getElementById('time-filter').value;
    
    // Get all goals
    const goals = window.allGoals || [];
    
    // Apply filters
    const filteredGoals = goals.filter(goal => {
        // Search term filter
        const matchesSearch = 
            goal.title.toLowerCase().includes(searchTerm) ||
            (goal.description && goal.description.toLowerCase().includes(searchTerm));
        
        // Status filter
        const matchesStatus = statusFilter === 'all' || 
            (statusFilter === 'in_progress' && !goal.completed) || 
            (statusFilter === 'completed' && goal.completed);
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || 
            (goal.subject_id && goal.subject_id === subjectFilter) || 
            (subjectFilter === 'none' && !goal.subject_id);
        
        // Time frame filter
        let matchesTimeFrame = true;
        const targetDate = new Date(goal.target_date);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        if (timeFilter === 'this_month') {
            const nextMonth = new Date(today);
            nextMonth.setMonth(nextMonth.getMonth() + 1);
            matchesTimeFrame = targetDate < nextMonth;
        } else if (timeFilter === 'this_semester') {
            // Assuming a semester is roughly 4 months
            const endOfSemester = new Date(today);
            endOfSemester.setMonth(endOfSemester.getMonth() + 4);
            matchesTimeFrame = targetDate < endOfSemester;
        } else if (timeFilter === 'this_year') {
            const endOfYear = new Date(today.getFullYear(), 11, 31);
            matchesTimeFrame = targetDate < endOfYear;
        }
        
        return matchesSearch && matchesStatus && matchesSubject && matchesTimeFrame;
    });
    
    // Display filtered goals using the stored subjects map
    if (window.subjectsMap) {
        displayGoals(filteredGoals, window.subjectsMap);
        
        // Update goal counts
        document.getElementById('in-progress-count').textContent = filteredGoals.filter(goal => !goal.completed).length;
        document.getElementById('completed-count').textContent = filteredGoals.filter(goal => goal.completed).length;
    } else {
        // Fallback to fetching subjects again if needed
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
            
            window.subjectsMap = subjectsMap;
            displayGoals(filteredGoals, subjectsMap);
            
            // Update goal counts
            document.getElementById('in-progress-count').textContent = filteredGoals.filter(goal => !goal.completed).length;
            document.getElementById('completed-count').textContent = filteredGoals.filter(goal => goal.completed).length;
        })
        .catch(error => {
            console.error('Error fetching subjects for filtering:', error);
        });
    }
}

// Open add goal modal
function openAddGoalModal() {
    // Reset form
    const goalForm = document.getElementById('goal-form');
    goalForm.reset();
    document.getElementById('goal-id').value = '';
    
    // Clear milestones
    document.getElementById('milestones-container').innerHTML = '';
    
    // Set default target date to 1 month from now
    const oneMonthLater = new Date();
    oneMonthLater.setMonth(oneMonthLater.getMonth() + 1);
    document.getElementById('target-date').value = formatDateForInput(oneMonthLater);
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Add New Goal';
    
    // Reset save button
    const saveBtn = document.getElementById('save-btn');
    saveBtn.textContent = 'Save Goal';
    saveBtn.disabled = false;
    
    // Show modal
    document.getElementById('goal-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit goal modal
function openEditGoalModal(goalId) {
    const token = localStorage.getItem('accessToken');
    
    // Reset form first
    const goalForm = document.getElementById('goal-form');
    goalForm.reset();
    
    // Clear milestones
    document.getElementById('milestones-container').innerHTML = '';
    
    // Reset save button
    const saveBtn = document.getElementById('save-btn');
    saveBtn.textContent = 'Update Goal';
    saveBtn.disabled = false;
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Edit Goal';
    
    // Show loading state in the modal
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    saveBtn.disabled = true;
    
    // Show modal immediately to show loading state
    document.getElementById('goal-modal').classList.add('active');
    document.body.classList.add('modal-open');
    
    // Fetch goal details
    fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch goal details');
        }
        return response.json();
    })
    .then(goal => {
        // Populate form
        document.getElementById('goal-id').value = goal._id;
        document.getElementById('title').value = goal.title;
        document.getElementById('description').value = goal.description || '';
        document.getElementById('target-date').value = formatDateForInput(new Date(goal.target_date));
        document.getElementById('subject-id').value = goal.subject_id || '';
        
        // Populate milestones
        const milestonesContainer = document.getElementById('milestones-container');
        milestonesContainer.innerHTML = '';
        
        if (goal.milestones && goal.milestones.length > 0) {
            goal.milestones.forEach((milestone, index) => {
                addMilestoneToForm(milestone.title, milestone.completed);
            });
        }
        
        // Reset save button
        saveBtn.textContent = 'Update Goal';
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error fetching goal details:', error);
        alert('Failed to load goal details. Please try again.');
        
        // Close modal on error
        closeModals();
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(goalId) {
    // Store goal ID for deletion
    const deleteBtn = document.getElementById('confirm-delete-btn');
    deleteBtn.dataset.id = goalId;
    
    // Reset delete button
    deleteBtn.textContent = 'Delete Goal';
    deleteBtn.disabled = false;
    
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
    
    // Reset goal form if it exists
    const goalForm = document.getElementById('goal-form');
    if (goalForm) {
        goalForm.reset();
        document.getElementById('goal-id').value = '';
        document.getElementById('milestones-container').innerHTML = '';
        
        // Reset save button
        const saveBtn = document.getElementById('save-btn');
        if (saveBtn) {
            saveBtn.textContent = 'Save Goal';
            saveBtn.disabled = false;
        }
    }
    
    // Reset delete button
    const deleteBtn = document.getElementById('confirm-delete-btn');
    if (deleteBtn) {
        deleteBtn.textContent = 'Delete Goal';
        deleteBtn.disabled = false;
        deleteBtn.dataset.id = '';
    }
}

// Add milestone to form
function addMilestone() {
    addMilestoneToForm('', false);
}

function addMilestoneToForm(title = '', completed = false) {
    const milestonesContainer = document.getElementById('milestones-container');
    const milestoneIndex = milestonesContainer.children.length;
    
    const milestoneItem = document.createElement('div');
    milestoneItem.className = 'milestone-form-item';
    
    milestoneItem.innerHTML = `
        <div class="milestone-input-group">
            <input type="text" name="milestone-${milestoneIndex}" placeholder="Milestone title" value="${title}" required>
            <div class="milestone-actions">
                <label class="milestone-checkbox-label">
                    <input type="checkbox" name="milestone-completed-${milestoneIndex}" ${completed ? 'checked' : ''}>
                    <span>Completed</span>
                </label>
                <button type="button" class="btn btn-icon remove-milestone-btn">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
    
    milestonesContainer.appendChild(milestoneItem);
    
    // Add event listener to remove button
    milestoneItem.querySelector('.remove-milestone-btn').addEventListener('click', function() {
        milestonesContainer.removeChild(milestoneItem);
    });
}

// Handle goal form submission
async function handleGoalFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const goalId = document.getElementById('goal-id').value;
    const isEdit = goalId !== '';
    
    // Get form data
    const title = document.getElementById('title').value;
    const description = document.getElementById('description').value;
    const targetDateStr = document.getElementById('target-date').value;
    const subjectId = document.getElementById('subject-id').value;
    
    // Validate required fields
    if (!title || !targetDateStr) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Replace these lines in your handleGoalFormSubmit function
    const targetDate = new Date(targetDateStr);
    targetDate.setHours(23, 59, 59, 999);

    // Instead of using toISOString(), use a format without timezone
    const formattedDate = `${targetDate.getFullYear()}-${String(targetDate.getMonth() + 1).padStart(2, '0')}-${String(targetDate.getDate()).padStart(2, '0')}T${String(targetDate.getHours()).padStart(2, '0')}:${String(targetDate.getMinutes()).padStart(2, '0')}:${String(targetDate.getSeconds()).padStart(2, '0')}`;

    // Now use formattedDate in your formData object
    
    // Prepare the form data object
    const formData = {
        title: title,
        description: description,
        target_date: formattedDate
    };
    
    // Add subject_id if selected
    if (subjectId) {
        formData.subject_id = subjectId;
    }
    
    // Get milestones
    formData.milestones = [];
    const milestonesContainer = document.getElementById('milestones-container');
    for (let i = 0; i < milestonesContainer.children.length; i++) {
        const milestoneItem = milestonesContainer.children[i];
        const titleInput = milestoneItem.querySelector('input[type="text"]');
        const completedCheckbox = milestoneItem.querySelector('input[type="checkbox"]');
        
        if (titleInput && titleInput.value.trim()) {
            formData.milestones.push({
                title: titleInput.value.trim(),
                completed: completedCheckbox ? completedCheckbox.checked : false
            });
        }
    }
    
    // Show loading state
    const saveBtn = document.getElementById('save-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        // Log what we're sending to help debug
        console.log('Sending goal data:', formData);
        
        let response;
        
        if (isEdit) {
            // Update existing goal
            response = await fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new goal
            response = await fetch('https://study-o5hp.onrender.com/goals', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        }
        
        // Check if response is ok
        if (!response.ok) {
            // Get detailed error information
            const errorText = await response.text();
            console.error('Server response:', response.status, errorText);
            
            try {
                const errorData = JSON.parse(errorText);
                throw new Error(errorData.detail || `Server error: ${response.status}`);
            } catch (jsonError) {
                throw new Error(`Server error: ${response.status} - ${errorText.substring(0, 100)}`);
            }
        }
        
        // Get the response data
        const goalData = await response.json();
        
        // Update local data if editing
        if (isEdit && window.allGoals) {
            window.allGoals = window.allGoals.map(goal => 
                goal._id === goalId ? goalData : goal
            );
        } else if (window.allGoals) {
            // Add new goal to local data
            window.allGoals.push(goalData);
        }
        
        // Update UI
        if (window.allGoals && window.subjectsMap) {
            updateGoalStats(window.allGoals);
            displayGoals(window.allGoals, window.subjectsMap);
        } else {
            // If local data is not available, reload all goals
            await loadGoals();
        }
        
        // Close modal
        closeModals();
        
        // Show success notification
        showNotification(isEdit ? 'Goal updated successfully' : 'Goal created successfully', 'success');
        
    } catch (error) {
        console.error('Error saving goal:', error);
        
        // Show error notification with details
        showNotification(`Error: ${error.message}`, 'error');
        
        // Reset button
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

// Update milestone completion status
async function updateMilestone(goalId, milestoneIndex, completed) {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Find the goal in our local data
        let goal;
        if (window.allGoals) {
            goal = window.allGoals.find(g => g._id === goalId);
        }
        
        if (!goal) {
            // Fetch current goal data if not in local data
            const response = await fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch goal data');
            }
            
            goal = await response.json();
        }
        
        // Check if milestone exists
        if (!goal.milestones || !goal.milestones[milestoneIndex]) {
            throw new Error('Milestone not found');
        }
        
        // Create a copy of the milestones array
        const updatedMilestones = [...goal.milestones];
        updatedMilestones[milestoneIndex] = {
            ...updatedMilestones[milestoneIndex],
            completed: completed
        };
        
        // Calculate new progress
        const totalMilestones = updatedMilestones.length;
        const completedMilestones = updatedMilestones.filter(m => m.completed).length;
        const progress = totalMilestones > 0 ? Math.round((completedMilestones / totalMilestones) * 100) : 0;
        
        // Update goal
        const updateResponse = await fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                milestones: updatedMilestones,
                progress: progress
            })
        });
        
        if (!updateResponse.ok) {
            throw new Error('Failed to update milestone');
        }
        
        // Get updated goal data
        const updatedGoal = await updateResponse.json();
        
        // Update local data
        if (window.allGoals) {
            window.allGoals = window.allGoals.map(g => 
                g._id === goalId ? updatedGoal : g
            );
        }
        
        // Update UI elements
        const progressBars = document.querySelectorAll(`.goal-card[data-id="${goalId}"] .progress-fill, .goal-item[data-id="${goalId}"] .progress-fill`);
        progressBars.forEach(bar => {
            bar.style.width = `${progress}%`;
        });
        
        const progressTexts = document.querySelectorAll(`.goal-card[data-id="${goalId}"] .progress-text, .goal-item[data-id="${goalId}"] .goal-progress-text`);
        progressTexts.forEach(text => {
            if (text.classList.contains('goal-progress-text')) {
                text.innerHTML = `<i class="fas fa-tasks"></i> ${progress}% Complete`;
            } else {
                text.textContent = `${progress}% Complete`;
            }
        });
        
        // If all milestones are completed, suggest marking the goal as complete
        if (progress === 100 && !updatedGoal.completed) {
            if (confirm('All milestones are completed! Would you like to mark this goal as complete?')) {
                markGoalComplete(goalId);
            }
        }
        
    } catch (error) {
        console.error('Error updating milestone:', error);
        showNotification(`Error: ${error.message}`, 'error');
        
        // Revert checkbox state
        const checkbox = document.querySelector(`input[data-goal-id="${goalId}"][data-index="${milestoneIndex}"]`);
        if (checkbox) {
            checkbox.checked = !completed;
        }
    }
}

// Mark goal as complete
async function markGoalComplete(goalId) {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading state on the button
        const completeButtons = document.querySelectorAll(`.mark-complete-btn[data-id="${goalId}"]`);
        completeButtons.forEach(button => {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Completing...';
        });
        
        const response = await fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                completed: true,
                completed_at: new Date().toISOString(),
                progress: 100
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to mark goal as complete');
        }
        
        // Get updated goal data
        const updatedGoal = await response.json();
        
        // Update local data
        if (window.allGoals) {
            window.allGoals = window.allGoals.map(goal => 
                goal._id === goalId ? updatedGoal : goal
            );
            
            // Update UI
            updateGoalStats(window.allGoals);
            displayGoals(window.allGoals, window.subjectsMap);
        } else {
            // If local data is not available, reload all goals
            await loadGoals();
        }
        
        // Show success notification
        showNotification('Goal marked as complete!', 'success');
        
    } catch (error) {
        console.error('Error marking goal as complete:', error);
        showNotification(`Error: ${error.message}`, 'error');
        
        // Reset button states
        const completeButtons = document.querySelectorAll(`.mark-complete-btn[data-id="${goalId}"]`);
        completeButtons.forEach(button => {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-check"></i> Mark Complete';
        });
    }
}

// Delete goal
async function deleteGoal() {
    const token = localStorage.getItem('accessToken');
    const deleteBtn = document.getElementById('confirm-delete-btn');
    const goalId = deleteBtn.dataset.id;
    
    if (!goalId) {
        console.error('No goal ID provided for deletion');
        closeModals();
        return;
    }
    
    // Show loading state
    const originalBtnText = deleteBtn.textContent;
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
    
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/goals/${goalId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete goal');
        }
        
        // Update local data
        if (window.allGoals) {
            window.allGoals = window.allGoals.filter(goal => goal._id !== goalId);
            
            // Update UI
            updateGoalStats(window.allGoals);
            displayGoals(window.allGoals, window.subjectsMap);
        } else {
            // If local data is not available, reload all goals
            await loadGoals();
        }
        
        // Close modal
        closeModals();
        
        // Show success notification
        showNotification('Goal deleted successfully', 'success');
        
    } catch (error) {
        console.error('Error deleting goal:', error);
        showNotification(`Error: ${error.message}`, 'error');
        
        // Reset button
        deleteBtn.disabled = false;
        deleteBtn.textContent = originalBtnText;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notification = document.querySelector('.notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.className = `notification ${type}`;
        document.body.appendChild(notification);
    } else {
        // Update existing notification
        notification.className = `notification ${type}`;
    }
    
    // Set notification content
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Show notification
    setTimeout(() => {
        notification.classList.add('active');
    }, 10);
    
    // Hide notification after a few seconds
    setTimeout(() => {
        notification.classList.remove('active');
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Toggle between kanban and list view
function toggleView() {
    const viewType = this.dataset.view;
    
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    this.classList.add('active');
    
    // Show selected view
    if (viewType === 'kanban') {
        document.getElementById('kanban-view').classList.add('view-active');
        document.getElementById('list-view').classList.remove('view-active');
    } else {
        document.getElementById('kanban-view').classList.remove('view-active');
        document.getElementById('list-view').classList.add('view-active');
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
}

function isDateNearing(date) {
    const today = new Date();
    const daysUntil = Math.ceil((date - today) / (1000 * 60 * 60 * 24));
    return daysUntil <= 7 && daysUntil >= 0; // Within a week
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