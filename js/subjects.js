// subjects.js - For the subjects page functionality

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
    
    // Load subjects data
    loadSubjects();
    
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
    // Add subject button
    document.getElementById('add-subject-btn').addEventListener('click', openAddSubjectModal);
    document.getElementById('empty-add-btn').addEventListener('click', openAddSubjectModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Subject form submission
    document.getElementById('subject-form').addEventListener('submit', handleSubjectFormSubmit);
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteSubject);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });
    
    // Search functionality
    document.getElementById('subject-search').addEventListener('input', debounce(filterSubjects, 300));
    
    // Color picker preview
    const colorInput = document.getElementById('color');
    const colorPreview = document.getElementById('color-preview');
    
    if (colorInput && colorPreview) {
        // Set initial preview color
        colorPreview.style.backgroundColor = colorInput.value;
        
        // Update preview on color change
        colorInput.addEventListener('input', function() {
            colorPreview.style.backgroundColor = this.value;
        });
    }
}

// Load subjects data
async function loadSubjects() {
    try {
        const token = localStorage.getItem('accessToken');
        
        // Show loading state
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'flex';
        });
        
        // Fetch all subjects
        const subjectsResponse = await fetch('https://study-o5hp.onrender.com/subjects', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!subjectsResponse.ok) {
            throw new Error('Failed to fetch subjects');
        }
        
        const subjects = await subjectsResponse.json();
        
        // Store subjects in a global variable for filtering
        window.allSubjects = subjects;
        
        // Display subjects
        displaySubjects(subjects);
        
    } catch (error) {
        console.error('Error loading subjects:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const subjectsGrid = document.getElementById('subjects-grid');
        subjectsGrid.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load subjects. Please try again later.</p>
            </div>
        `;
    }
}

// Display subjects
function displaySubjects(subjects) {
    const subjectsGrid = document.getElementById('subjects-grid');
    const subjectsList = document.getElementById('subjects-list');
    const noSubjects = document.getElementById('no-subjects');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Update counts
    document.getElementById('total-subjects-count').textContent = window.allSubjects.length;
    document.getElementById('shown-subjects-count').textContent = subjects.length;
    
    // Check if there are any subjects
    if (subjects.length === 0) {
        subjectsGrid.innerHTML = '';
        subjectsList.innerHTML = '';
        noSubjects.style.display = 'block';
        return;
    }
    
    // Hide no subjects message
    noSubjects.style.display = 'none';
    
    // Clear existing subjects
    subjectsGrid.innerHTML = '';
    subjectsList.innerHTML = '';
    
    // Add subjects to grid and list views
    subjects.forEach(subject => {
        // Create grid item
        const gridItem = document.createElement('div');
        gridItem.className = 'subject-card';
        gridItem.dataset.id = subject._id;
        
        gridItem.innerHTML = `
            <div class="subject-color" style="background-color: ${subject.color}"></div>
            <div class="subject-content">
                <h3>${subject.name}</h3>
                <p>${subject.description || 'No description provided'}</p>
                <div class="subject-actions">
                    <button class="btn btn-icon edit-btn" data-id="${subject._id}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${subject._id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Create list item
        const listItem = document.createElement('div');
        listItem.className = 'subject-item';
        listItem.dataset.id = subject._id;
        
        listItem.innerHTML = `
            <div class="subject-color-indicator" style="background-color: ${subject.color}"></div>
            <div class="subject-content">
                <div class="subject-header">
                    <h3>${subject.name}</h3>
                    <div class="subject-actions">
                        <button class="btn btn-icon edit-btn" data-id="${subject._id}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon delete-btn" data-id="${subject._id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <p>${subject.description || 'No description provided'}</p>
            </div>
        `;
        
        // Add to containers
        subjectsGrid.appendChild(gridItem);
        subjectsList.appendChild(listItem);
    });
    
    // Add event listeners to edit and delete buttons
    document.querySelectorAll('.edit-btn').forEach(button => {
        button.addEventListener('click', function() {
            const subjectId = this.dataset.id;
            openEditSubjectModal(subjectId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', function() {
            const subjectId = this.dataset.id;
            openDeleteConfirmationModal(subjectId);
        });
    });
}

// Filter subjects
function filterSubjects() {
    const searchTerm = document.getElementById('subject-search').value.toLowerCase();
    
    // Get all subjects
    const subjects = window.allSubjects || [];
    
    // Apply filters
    const filteredSubjects = subjects.filter(subject => {
        // Search term filter
        const matchesSearch = 
            subject.name.toLowerCase().includes(searchTerm) ||
            (subject.description && subject.description.toLowerCase().includes(searchTerm));
        
        return matchesSearch;
    });
    
    // Display filtered subjects
    displaySubjects(filteredSubjects);
}

// Open add subject modal
function openAddSubjectModal() {
    // Reset form
    const subjectForm = document.getElementById('subject-form');
    subjectForm.reset();
    document.getElementById('subject-id').value = '';
    
    // Set default color
    document.getElementById('color').value = '#4287f5';
    document.getElementById('color-preview').style.backgroundColor = '#4287f5';
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Add New Subject';
    
    // Reset save button
    const saveBtn = document.getElementById('save-btn');
    saveBtn.textContent = 'Save Subject';
    saveBtn.disabled = false;
    
    // Show modal
    document.getElementById('subject-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit subject modal
function openEditSubjectModal(subjectId) {
    const token = localStorage.getItem('accessToken');
    
    // Reset form first
    const subjectForm = document.getElementById('subject-form');
    subjectForm.reset();
    
    // Reset save button
    const saveBtn = document.getElementById('save-btn');
    saveBtn.textContent = 'Update Subject';
    saveBtn.disabled = false;
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Edit Subject';
    
    // Show loading state in the modal
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    saveBtn.disabled = true;
    
    // Show modal immediately to show loading state
    document.getElementById('subject-modal').classList.add('active');
    document.body.classList.add('modal-open');
    
    // Fetch subject details
    fetch(`https://study-o5hp.onrender.com/subjects/${subjectId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch subject details');
        }
        return response.json();
    })
    .then(subject => {
        // Populate form
        document.getElementById('subject-id').value = subject._id;
        document.getElementById('name').value = subject.name;
        document.getElementById('description').value = subject.description || '';
        document.getElementById('color').value = subject.color;
        document.getElementById('color-preview').style.backgroundColor = subject.color;
        
        // Reset save button
        saveBtn.textContent = 'Update Subject';
        saveBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error fetching subject details:', error);
        alert('Failed to load subject details. Please try again.');
        
        // Close modal on error
        closeModals();
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(subjectId) {
    // Store subject ID for deletion
    const deleteBtn = document.getElementById('confirm-delete-btn');
    deleteBtn.dataset.id = subjectId;
    
    // Reset delete button
    deleteBtn.textContent = 'Delete Subject';
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
    
    // Reset subject form if it exists
    const subjectForm = document.getElementById('subject-form');
    if (subjectForm) {
        subjectForm.reset();
        document.getElementById('subject-id').value = '';
        
        // Reset save button
        const saveBtn = document.getElementById('save-btn');
        if (saveBtn) {
            saveBtn.textContent = 'Save Subject';
            saveBtn.disabled = false;
        }
    }
    
    // Reset delete button
    const deleteBtn = document.getElementById('confirm-delete-btn');
    if (deleteBtn) {
        deleteBtn.textContent = 'Delete Subject';
        deleteBtn.disabled = false;
        deleteBtn.dataset.id = '';
    }
}

// Handle subject form submission
async function handleSubjectFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const subjectId = document.getElementById('subject-id').value;
    const isEdit = subjectId !== '';
    
    // Get form data
    const formData = {
        name: document.getElementById('name').value,
        description: document.getElementById('description').value,
        color: document.getElementById('color').value
    };
    
    // Validate form data
    if (!formData.name || formData.name.trim() === '') {
        alert('Subject name is required');
        return;
    }
    
    // Show loading state
    const saveBtn = document.getElementById('save-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        let response;
        
        if (isEdit) {
            // Update existing subject
            response = await fetch(`https://study-o5hp.onrender.com/subjects/${subjectId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new subject
            response = await fetch('https://study-o5hp.onrender.com/subjects', {
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
            throw new Error(errorData.detail || 'Failed to save subject');
        }
        
        // Get the response data
        const subjectData = await response.json();
        
        // Update local data if editing
        if (isEdit && window.allSubjects) {
            window.allSubjects = window.allSubjects.map(subject => 
                subject._id === subjectId ? subjectData : subject
            );
            displaySubjects(window.allSubjects);
        } else if (window.allSubjects) {
            // Add new subject to local data
            window.allSubjects.push(subjectData);
            displaySubjects(window.allSubjects);
        } else {
            // Reload all subjects if local data doesn't exist
            await loadSubjects();
        }
        
        // Close modal and reset form
        closeModals();
    } catch (error) {
        console.error('Error saving subject:', error);
        alert(error.message);
        
        // Reset button
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

// Delete subject
async function deleteSubject() {
    const token = localStorage.getItem('accessToken');
    const deleteBtn = document.getElementById('confirm-delete-btn');
    const subjectId = deleteBtn.dataset.id;
    
    if (!subjectId) {
        console.error('No subject ID provided for deletion');
        closeModals();
        return;
    }
    
    // Show loading state
    const originalBtnText = deleteBtn.textContent;
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
    
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/subjects/${subjectId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete subject');
        }
        
        // Update local data
        if (window.allSubjects) {
            window.allSubjects = window.allSubjects.filter(subject => subject._id !== subjectId);
            displaySubjects(window.allSubjects);
        } else {
            // Reload all subjects if local data doesn't exist
            await loadSubjects();
        }
        
        // Close modal and reset button
        closeModals();
    } catch (error) {
        console.error('Error deleting subject:', error);
        alert(error.message);
        
        // Reset button
        deleteBtn.disabled = false;
        deleteBtn.textContent = originalBtnText;
    }
}

// Toggle between grid and list view
function toggleView() {
    const viewType = this.dataset.view;
    
    // Update active button
    document.querySelectorAll('.view-toggle-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    this.classList.add('active');
    
    // Show selected view
    if (viewType === 'grid') {
        document.getElementById('subjects-grid').classList.add('view-active');
        document.getElementById('subjects-list').classList.remove('view-active');
    } else {
        document.getElementById('subjects-grid').classList.remove('view-active');
        document.getElementById('subjects-list').classList.add('view-active');
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