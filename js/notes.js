// notes.js - For the notes page functionality

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
    
    // Load notes data
    loadNotes();
    
    // Initialize event listeners
    initEventListeners();
    
    // Check URL for new note parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('new') === 'true') {
        openAddNoteModal();
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
    // Add note button
    document.getElementById('add-note-btn').addEventListener('click', openAddNoteModal);
    document.getElementById('empty-add-btn').addEventListener('click', openAddNoteModal);
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Cancel buttons
    document.getElementById('cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Note form submission
    const noteForm = document.getElementById('note-form');
    // Remove any existing event listeners
    const newNoteForm = noteForm.cloneNode(true);
    noteForm.parentNode.replaceChild(newNoteForm, noteForm);
    newNoteForm.addEventListener('submit', handleNoteFormSubmit);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });
    
    // Search functionality
    const searchInput = document.getElementById('note-search');
    // Remove any existing event listeners
    const newSearchInput = searchInput.cloneNode(true);
    searchInput.parentNode.replaceChild(newSearchInput, searchInput);
    newSearchInput.addEventListener('input', debounce(filterNotes, 300));
    
    // Filters
    const subjectFilter = document.getElementById('subject-filter');
    const newSubjectFilter = subjectFilter.cloneNode(true);
    subjectFilter.parentNode.replaceChild(newSubjectFilter, subjectFilter);
    newSubjectFilter.addEventListener('change', filterNotes);
    
    const sortFilter = document.getElementById('sort-filter');
    const newSortFilter = sortFilter.cloneNode(true);
    sortFilter.parentNode.replaceChild(newSortFilter, sortFilter);
    newSortFilter.addEventListener('change', filterNotes);
}

// Load notes data
async function loadNotes() {
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
        
        // Populate subject filters
        const subjectFilter = document.getElementById('subject-filter');
        const subjectSelect = document.getElementById('subject-id');
        
        // Clear existing options except the first one
        while (subjectFilter.options.length > 1) {
            subjectFilter.remove(1);
        }
        
        while (subjectSelect.options.length > 1) {
            subjectSelect.remove(1);
        }
        
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
        
        // Fetch all notes
        const notesResponse = await fetch('https://study-o5hp.onrender.com/notes', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!notesResponse.ok) {
            throw new Error('Failed to fetch notes');
        }
        
        const notes = await notesResponse.json();
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store notes in a global variable for filtering
        window.allNotes = notes;
        
        // Display notes
        displayNotes(notes, subjectsMap);
        
        // Extract and display tags
        displayTags(notes);
        
    } catch (error) {
        console.error('Error loading notes:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const notesGrid = document.getElementById('notes-grid');
        notesGrid.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load notes. Please try again later.</p>
            </div>
        `;
    }
}

// Display notes
function displayNotes(notes, subjectsMap) {
    const notesGrid = document.getElementById('notes-grid');
    const notesList = document.getElementById('notes-list');
    const noNotes = document.getElementById('no-notes');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Update counts
    document.getElementById('total-notes-count').textContent = window.allNotes.length;
    document.getElementById('shown-notes-count').textContent = notes.length;
    
    // Check if there are any notes
    if (notes.length === 0) {
        notesGrid.innerHTML = '';
        notesList.innerHTML = '';
        noNotes.style.display = 'block';
        return;
    }
    
    // Hide no notes message
    noNotes.style.display = 'none';
    
    // Clear existing notes
    notesGrid.innerHTML = '';
    notesList.innerHTML = '';
    
    // Add notes to grid and list views
    notes.forEach(note => {
        const subject = note.subject_id ? (subjectsMap[note.subject_id] || { name: 'No Subject', color: '#808080' }) : { name: 'No Subject', color: '#808080' };
        const createdDate = new Date(note.created_at);
        const updatedDate = note.updated_at ? new Date(note.updated_at) : null;
        
        // Create grid item
        const gridItem = document.createElement('div');
        gridItem.className = 'note-card';
        gridItem.dataset.id = note._id;
        
        // Generate random pastel color if no subject
        const cardColor = subject.color !== '#808080' ? subject.color : getRandomPastelColor();
        
        gridItem.innerHTML = `
            <div class="card-header" style="background-color: ${cardColor}">
                <h3>${note.title}</h3>
                <div class="card-actions">
                    <button class="btn btn-icon edit-btn" data-id="${note._id}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${note._id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="note-content">${formatNoteContent(note.content)}</div>
                <div class="note-meta">
                    <div class="meta-item">
                        <i class="fas fa-book"></i>
                        <span>${subject.name}</span>
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-calendar-alt"></i>
                        <span>${formatDate(updatedDate || createdDate)}</span>
                    </div>
                </div>
                ${note.tags.length > 0 ? `
                <div class="note-tags">
                    ${note.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                ` : ''}
            </div>
        `;
        
        // Create list item
        const listItem = document.createElement('div');
        listItem.className = 'note-item';
        listItem.dataset.id = note._id;
        
        listItem.innerHTML = `
            <div class="note-header">
                <h3>${note.title}</h3>
                <div class="note-actions">
                    <button class="btn btn-icon edit-btn" data-id="${note._id}">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${note._id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="note-preview">${formatNoteContent(note.content, true)}</div>
            <div class="note-meta">
                <span class="subject-tag" style="background-color: ${subject.color}">
                    ${subject.name}
                </span>
                <span class="note-date">
                    <i class="fas fa-calendar-alt"></i>
                    ${formatDate(updatedDate || createdDate)}
                </span>
                ${note.tags.length > 0 ? `
                <div class="note-tags">
                    ${note.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                ` : ''}
            </div>
        `;
        
        // Add to containers
        notesGrid.appendChild(gridItem);
        notesList.appendChild(listItem);
        
        // Add click event to open note
        gridItem.addEventListener('click', function(e) {
            // Don't open note if clicking on action buttons
            if (!e.target.closest('.card-actions')) {
                openNoteViewModal(note._id);
            }
        });
        
        listItem.addEventListener('click', function(e) {
            // Don't open note if clicking on action buttons
            if (!e.target.closest('.note-actions')) {
                openNoteViewModal(note._id);
            }
        });
    });
    
    // Add event listeners to edit and delete buttons
    document.querySelectorAll('.edit-btn').forEach(button => {
        // Remove any existing event listeners
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
        
        newButton.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent note opening
            const noteId = this.dataset.id;
            openEditNoteModal(noteId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(button => {
        // Remove any existing event listeners
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
        
        newButton.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent note opening
            const noteId = this.dataset.id;
            openDeleteConfirmationModal(noteId);
        });
    });
}

// Display tags
function displayTags(notes) {
    const tagsContainer = document.getElementById('tags-container');
    tagsContainer.innerHTML = '';
    
    // Extract all tags
    const allTags = {};
    notes.forEach(note => {
        note.tags.forEach(tag => {
            if (allTags[tag]) {
                allTags[tag]++;
            } else {
                allTags[tag] = 1;
            }
        });
    });
    
    // Sort tags by frequency
    const sortedTags = Object.keys(allTags).sort((a, b) => allTags[b] - allTags[a]);
    
    // Display tags
    sortedTags.forEach(tag => {
        const tagElement = document.createElement('div');
        tagElement.className = 'tag';
        tagElement.innerHTML = `
            <span class="tag-name">${tag}</span>
            <span class="tag-count">${allTags[tag]}</span>
        `;
        
        // Add click event to filter by tag
        tagElement.addEventListener('click', function() {
            filterNotesByTag(tag);
        });
        
        tagsContainer.appendChild(tagElement);
    });
    
    // Add "All Tags" option if there are tags
    if (sortedTags.length > 0) {
        const allTagsElement = document.createElement('div');
        allTagsElement.className = 'tag all-tags';
        allTagsElement.innerHTML = `
            <span class="tag-name">All Tags</span>
        `;
        
        // Add click event to clear tag filter
        allTagsElement.addEventListener('click', function() {
            filterNotes();
        });
        
        tagsContainer.insertBefore(allTagsElement, tagsContainer.firstChild);
    }
}

// Filter notes by tag
function filterNotesByTag(tag) {
    // Update active tag
    document.querySelectorAll('.tag').forEach(tagElement => {
        tagElement.classList.remove('active');
        if (tagElement.querySelector('.tag-name').textContent === tag) {
            tagElement.classList.add('active');
        }
    });
    
    // Filter notes
    const token = localStorage.getItem('accessToken');
    
    fetch('https://study-o5hp.onrender.com/notes?tag=' + encodeURIComponent(tag), {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(notes => {
        // Update global notes for further filtering
        window.filteredByTag = notes;
        
        // Apply other filters
        filterNotes(true);
    })
    .catch(error => {
        console.error('Error filtering notes by tag:', error);
    });
}

// Filter notes
function filterNotes(keepTagFilter = false) {
    const searchTerm = document.getElementById('note-search').value.toLowerCase();
    const subjectFilter = document.getElementById('subject-filter').value;
    const sortFilter = document.getElementById('sort-filter').value;
    
    // Get notes (either all notes or those filtered by tag)
    const notes = keepTagFilter && window.filteredByTag ? window.filteredByTag : window.allNotes || [];
    
    // If not keeping tag filter, remove active class from tags
    if (!keepTagFilter) {
        document.querySelectorAll('.tag').forEach(tagElement => {
            tagElement.classList.remove('active');
        });
        
        // Add active class to "All Tags" if it exists
        const allTagsElement = document.querySelector('.tag.all-tags');
        if (allTagsElement) {
            allTagsElement.classList.add('active');
        }
    }
    
    // Apply filters
    let filteredNotes = notes.filter(note => {
        // Search term filter
        const matchesSearch = 
            note.title.toLowerCase().includes(searchTerm) ||
            note.content.toLowerCase().includes(searchTerm) ||
            note.tags.some(tag => tag.toLowerCase().includes(searchTerm));
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || 
            (note.subject_id && note.subject_id === subjectFilter) || 
            (subjectFilter === 'none' && !note.subject_id);
        
        return matchesSearch && matchesSubject;
    });
    
    // Apply sorting
    switch (sortFilter) {
        case 'newest':
            filteredNotes.sort((a, b) => {
                const dateA = a.updated_at ? new Date(a.updated_at) : new Date(a.created_at);
                const dateB = b.updated_at ? new Date(b.updated_at) : new Date(b.created_at);
                return dateB - dateA;
            });
            break;
        case 'oldest':
            filteredNotes.sort((a, b) => {
                const dateA = a.updated_at ? new Date(a.updated_at) : new Date(a.created_at);
                const dateB = b.updated_at ? new Date(b.updated_at) : new Date(b.created_at);
                return dateA - dateB;
            });
            break;
        case 'title_asc':
            filteredNotes.sort((a, b) => a.title.localeCompare(b.title));
            break;
        case 'title_desc':
            filteredNotes.sort((a, b) => b.title.localeCompare(a.title));
            break;
    }
    
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
        
        // Display filtered notes
        displayNotes(filteredNotes, subjectsMap);
    })
    .catch(error => {
        console.error('Error fetching subjects for filtering:', error);
    });
}

// Open add note modal
function openAddNoteModal() {
    // Reset form
    document.getElementById('note-form').reset();
    document.getElementById('note-id').value = '';
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Add New Note';
    document.getElementById('save-btn').textContent = 'Save Note';
    
    // Show modal
    document.getElementById('note-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open edit note modal
function openEditNoteModal(noteId) {
    const token = localStorage.getItem('accessToken');
    
    // Try to find note in local data first
    const note = window.allNotes?.find(n => n._id === noteId);
    
    if (note) {
        populateEditForm(note);
    } else {
        // Fetch note details from API if not found locally
        fetch(`https://study-o5hp.onrender.com/notes/${noteId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => response.json())
        .then(note => {
            populateEditForm(note);
        })
        .catch(error => {
            console.error('Error fetching note details:', error);
            alert('Failed to load note details. Please try again.');
        });
    }
}

// Helper function to populate edit form
function populateEditForm(note) {
    // Populate form
    document.getElementById('note-id').value = note._id;
    document.getElementById('title').value = note.title;
    document.getElementById('content').value = note.content;
    document.getElementById('subject-id').value = note.subject_id || '';
    document.getElementById('tags').value = note.tags.join(', ');
    
    // Update modal title
    document.getElementById('modal-title').textContent = 'Edit Note';
    document.getElementById('save-btn').textContent = 'Update Note';
    
    // Show modal
    document.getElementById('note-modal').classList.add('active');
    document.body.classList.add('modal-open');
}

// Open note view modal
function openNoteViewModal(noteId) {
    const token = localStorage.getItem('accessToken');
    
    // Find note in local data first to avoid unnecessary API call
    const note = window.allNotes?.find(n => n._id === noteId);
    
    if (note) {
        displayNoteInViewModal(note);
    } else {
        // Fetch note details from API if not found locally
        fetch(`https://study-o5hp.onrender.com/notes/${noteId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => response.json())
        .then(note => {
            displayNoteInViewModal(note);
        })
        .catch(error => {
            console.error('Error fetching note details:', error);
            alert('Failed to load note details. Please try again.');
        });
    }
}

// Helper function to display note in view modal
function displayNoteInViewModal(note) {
    const token = localStorage.getItem('accessToken');
    
    // Fetch subject info
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
        
        const subject = note.subject_id ? (subjectsMap[note.subject_id] || { name: 'No Subject' }) : { name: 'No Subject' };
        const noteDate = note.updated_at ? new Date(note.updated_at) : new Date(note.created_at);
        
        // Populate modal
        document.getElementById('view-title').textContent = note.title;
        document.getElementById('view-subject').textContent = subject.name;
        document.getElementById('view-date').textContent = formatDate(noteDate);
        
        // Format content with line breaks and links
        document.getElementById('view-content').innerHTML = formatNoteContent(note.content);
        
        // Display tags
        const tagsContainer = document.getElementById('view-tags');
        tagsContainer.innerHTML = '';
        
        if (note.tags.length > 0) {
            note.tags.forEach(tag => {
                const tagElement = document.createElement('span');
                tagElement.className = 'tag';
                tagElement.textContent = tag;
                tagsContainer.appendChild(tagElement);
            });
        } else {
            tagsContainer.innerHTML = '<span class="no-tags">No tags</span>';
        }
        
        // Set up action buttons
        document.getElementById('edit-note-btn').dataset.id = note._id;
        document.getElementById('delete-note-btn').dataset.id = note._id;
        
        // Remove existing event listeners
        const editBtn = document.getElementById('edit-note-btn');
        const newEditBtn = editBtn.cloneNode(true);
        editBtn.parentNode.replaceChild(newEditBtn, editBtn);
        
        const deleteBtn = document.getElementById('delete-note-btn');
        const newDeleteBtn = deleteBtn.cloneNode(true);
        deleteBtn.parentNode.replaceChild(newDeleteBtn, deleteBtn);
        
        // Show modal
        document.getElementById('note-view-modal').classList.add('active');
        document.body.classList.add('modal-open');
        
        // Add event listeners
        newEditBtn.addEventListener('click', function() {
            closeModals();
            openEditNoteModal(note._id);
        });
        
        newDeleteBtn.addEventListener('click', function() {
            closeModals();
            openDeleteConfirmationModal(note._id);
        });
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(noteId) {
    // Store note ID for deletion
    document.getElementById('confirm-delete-btn').dataset.id = noteId;
    
    // Remove existing event listeners
    const deleteBtn = document.getElementById('confirm-delete-btn');
    const newDeleteBtn = deleteBtn.cloneNode(true);
    deleteBtn.parentNode.replaceChild(newDeleteBtn, deleteBtn);
    
    // Add event listener
    newDeleteBtn.addEventListener('click', deleteNote);
    
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

// Handle note form submission
async function handleNoteFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const noteId = document.getElementById('note-id').value;
    const isEdit = noteId !== '';
    const saveBtn = document.getElementById('save-btn');
    
    // Get form data
    const formData = {
        title: document.getElementById('title').value,
        content: document.getElementById('content').value,
        tags: document.getElementById('tags').value.split(',').map(tag => tag.trim()).filter(tag => tag)
    };
    
    // Add subject_id if selected
    const subjectId = document.getElementById('subject-id').value;
    if (subjectId) {
        formData.subject_id = subjectId;
    }
    
    try {
        // Show loading state
        const originalBtnText = saveBtn.textContent;
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
        
        let response;
        
        if (isEdit) {
            // Update existing note
            response = await fetch(`https://study-o5hp.onrender.com/notes/${noteId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(formData)
            });
        } else {
            // Create new note
            response = await fetch('https://study-o5hp.onrender.com/notes', {
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
            throw new Error(errorData.detail || 'Failed to save note');
        }
        
        const savedNote = await response.json();
        
        // Update local data without reloading
        if (isEdit) {
            // Update in allNotes
            if (window.allNotes) {
                window.allNotes = window.allNotes.map(note => 
                    note._id === savedNote._id ? savedNote : note
                );
            }
            
            // Update in filteredByTag if it exists
            if (window.filteredByTag) {
                window.filteredByTag = window.filteredByTag.map(note => 
                    note._id === savedNote._id ? savedNote : note
                );
            }
        } else {
            // Add to allNotes
            if (window.allNotes) {
                window.allNotes.unshift(savedNote);
            } else {
                window.allNotes = [savedNote];
            }
            
            // Add to filteredByTag if it exists and matches the tag
            if (window.filteredByTag && savedNote.tags.some(tag => 
                document.querySelector('.tag.active .tag-name')?.textContent === tag)) {
                window.filteredByTag.unshift(savedNote);
            }
        }
        
        // Close modal and refresh display
        closeModals();
        filterNotes();
        
    } catch (error) {
        console.error('Error saving note:', error);
        alert(error.message);
    } finally {
        // Reset button
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText || 'Save Note';
    }
}

// Delete note
async function deleteNote() {
    const token = localStorage.getItem('accessToken');
    const noteId = document.getElementById('confirm-delete-btn').dataset.id;
    
    try {
        // Show loading state
        const deleteBtn = document.getElementById('confirm-delete-btn');
        const originalBtnText = deleteBtn.textContent;
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
        
        const response = await fetch(`https://study-o5hp.onrender.com/notes/${noteId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete note');
        }
        
        // Remove the deleted note from allNotes array
        if (window.allNotes) {
            window.allNotes = window.allNotes.filter(note => note._id !== noteId);
        }
        
        // Also remove from filteredByTag if it exists
        if (window.filteredByTag) {
            window.filteredByTag = window.filteredByTag.filter(note => note._id !== noteId);
        }
        
        // Close modal and refresh display
        closeModals();
        filterNotes();
        
    } catch (error) {
        console.error('Error deleting note:', error);
        alert(error.message);
    } finally {
        // Reset button
        const deleteBtn = document.getElementById('confirm-delete-btn');
        if (deleteBtn) {
            deleteBtn.disabled = false;
            deleteBtn.textContent = 'Delete';
        }
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
        document.getElementById('notes-grid').classList.add('view-active');
        document.getElementById('notes-list').classList.remove('view-active');
    } else {
        document.getElementById('notes-grid').classList.remove('view-active');
        document.getElementById('notes-list').classList.add('view-active');
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatNoteContent(content, truncate = false) {
    if (!content) return '';
    
    // Convert line breaks to <br> tags
    let formattedContent = content.replace(/\n/g, '<br>');
    
    // Convert URLs to clickable links
    formattedContent = formattedContent.replace(
        /(https?:\/\/[^\s]+)/g, 
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    
    // Truncate if needed
    if (truncate && formattedContent.length > 150) {
        formattedContent = formattedContent.substring(0, 150) + '...';
    }
    
    return formattedContent;
}

function getRandomPastelColor() {
    // Generate pastel colors for notes without subjects
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue}, 70%, 80%)`;
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
        