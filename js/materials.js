// materials.js - For the materials page functionality

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
    
    // Load materials data
    loadMaterials();
    
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
    // Upload material button
    document.getElementById('upload-material-btn').addEventListener('click', openUploadModal);
    document.getElementById('empty-upload-btn').addEventListener('click', openUploadModal);
    
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
    document.getElementById('edit-cancel-btn').addEventListener('click', closeModals);
    document.getElementById('delete-cancel-btn').addEventListener('click', closeModals);
    
    // Upload form submission
    document.getElementById('upload-form').addEventListener('submit', handleUploadFormSubmit);
    
    // Edit material form submission
    document.getElementById('edit-material-form').addEventListener('submit', handleEditMaterialFormSubmit);
    
    // Delete confirmation
    document.getElementById('confirm-delete-btn').addEventListener('click', deleteMaterial);
    
    // View toggle buttons
    document.querySelectorAll('.view-toggle-btn').forEach(button => {
        button.addEventListener('click', toggleView);
    });
    
    // Search functionality
    document.getElementById('material-search').addEventListener('input', debounce(filterMaterials, 300));
    
    // Filters
    document.getElementById('file-type-filter').addEventListener('change', filterMaterials);
    document.getElementById('subject-filter').addEventListener('change', filterMaterials);
    document.getElementById('sort-filter').addEventListener('change', filterMaterials);
    
    // File upload area
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const browseLink = document.querySelector('.browse-link');
    
    if (uploadArea && fileInput) {
        // Handle drag and drop events
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelection(fileInput.files[0]);
            }
        });
        
        // Handle click to browse
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
        
        // Handle file selection
        fileInput.addEventListener('change', function() {
            if (this.files.length) {
                handleFileSelection(this.files[0]);
            }
        });
        
        // Handle remove file button
        document.getElementById('remove-file-btn').addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            fileInput.value = '';
            document.getElementById('file-preview').style.display = 'none';
            uploadArea.style.display = 'flex';
        });
    }
}

// Handle file selection for upload
function handleFileSelection(file) {
    const filePreview = document.getElementById('file-preview');
    const uploadArea = document.getElementById('upload-area');
    const fileNameElement = document.getElementById('preview-file-name');
    const fileSizeElement = document.getElementById('preview-file-size');
    const fileIcon = document.querySelector('.file-icon');
    
    // Check file size
    if (file.size > 5 * 1024 * 1024) { // 5MB
        alert('File size exceeds the maximum limit of 5MB');
        return;
    }
    
    // Check file type
    const fileExt = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'xlsx', 'xls'];
    
    if (!allowedExts.includes(fileExt)) {
        alert('File type not allowed. Allowed types: PDF, DOC, DOCX, TXT, JPG, PNG, XLSX, XLS');
        return;
    }
    
    // Update file preview
    fileNameElement.textContent = file.name;
    fileSizeElement.textContent = formatFileSize(file.size);
    
    // Set appropriate icon
    if (['pdf'].includes(fileExt)) {
        fileIcon.className = 'fas fa-file-pdf file-icon';
    } else if (['doc', 'docx'].includes(fileExt)) {
        fileIcon.className = 'fas fa-file-word file-icon';
    } else if (['txt'].includes(fileExt)) {
        fileIcon.className = 'fas fa-file-alt file-icon';
    } else if (['jpg', 'jpeg', 'png'].includes(fileExt)) {
        fileIcon.className = 'fas fa-file-image file-icon';
    } else if (['xlsx', 'xls'].includes(fileExt)) {
        fileIcon.className = 'fas fa-file-excel file-icon';
    } else {
        fileIcon.className = 'fas fa-file file-icon';
    }
    
    // Show preview and hide upload area
    filePreview.style.display = 'flex';
    uploadArea.style.display = 'none';
    
    // Set default material name from filename
    const materialName = document.getElementById('name');
    if (materialName && !materialName.value) {
        materialName.value = file.name.split('.')[0];
    }
}

// Load materials data
async function loadMaterials() {
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
        const editSubjectSelect = document.getElementById('edit-subject-id');
        
        // Keep only the first option
        while (subjectFilter.options.length > 1) {
            subjectFilter.remove(1);
        }
        
        while (subjectSelect.options.length > 1) {
            subjectSelect.remove(1);
        }
        
        while (editSubjectSelect.options.length > 1) {
            editSubjectSelect.remove(1);
        }
        
        // Populate subject filters
        subjects.forEach(subject => {
            // Add to filter
            const filterOption = document.createElement('option');
            filterOption.value = subject._id;
            filterOption.textContent = subject.name;
            subjectFilter.appendChild(filterOption);
            
            // Add to upload form select
            const selectOption = document.createElement('option');
            selectOption.value = subject._id;
            selectOption.textContent = subject.name;
            subjectSelect.appendChild(selectOption);
            
            // Add to edit form select
            const editSelectOption = document.createElement('option');
            editSelectOption.value = subject._id;
            editSelectOption.textContent = subject.name;
            editSubjectSelect.appendChild(editSelectOption);
        });
        
        // Fetch all materials
        const materialsResponse = await fetch('https://study-o5hp.onrender.com/materials', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!materialsResponse.ok) {
            throw new Error('Failed to fetch materials');
        }
        
        const materials = await materialsResponse.json();
        
        // Create a map of subjects for easy lookup
        const subjectsMap = subjects.reduce((acc, subject) => {
            acc[subject._id] = subject;
            return acc;
        }, {});
        
        // Store materials and subjects in global variables for filtering
        window.allMaterials = materials;
        window.subjectsMap = subjectsMap;
        
        // Display materials
        displayMaterials(materials, subjectsMap);
        
        // Update storage info
        updateStorageInfo(materials);
        
    } catch (error) {
        console.error('Error loading materials:', error);
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
        
        // Show error message
        const materialsGrid = document.getElementById('materials-grid');
        materialsGrid.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>Failed to load materials. Please try again later.</p>
            </div>
        `;
    }
}

// Display materials
function displayMaterials(materials, subjectsMap) {
    const materialsGrid = document.getElementById('materials-grid');
    const materialsList = document.getElementById('materials-list');
    const noMaterials = document.getElementById('no-materials');
    
    // Hide loading spinners
    document.querySelectorAll('.loading-spinner').forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // Update counts
    document.getElementById('total-materials-count').textContent = window.allMaterials.length;
    document.getElementById('shown-materials-count').textContent = materials.length;
    
    // Check if there are any materials
    if (materials.length === 0) {
        materialsGrid.innerHTML = '';
        materialsList.innerHTML = '';
        noMaterials.style.display = 'block';
        return;
    }
    
    // Hide no materials message
    noMaterials.style.display = 'none';
    
    // Clear existing materials
    materialsGrid.innerHTML = '';
    materialsList.innerHTML = '';
    
    // Add materials to grid and list views
    materials.forEach(material => {
        const subject = subjectsMap[material.subject_id] || { name: 'No Subject', color: '#808080' };
        const uploadDate = new Date(material.uploaded_at);
        
        // Create grid item
        const gridItem = document.createElement('div');
        gridItem.className = 'material-card';
        gridItem.dataset.id = material._id;
        
        gridItem.innerHTML = `
            <div class="material-icon ${material.file_type}">
                <i class="fas ${getFileIcon(material.file_type)}"></i>
            </div>
            <div class="material-content">
                <h3>${material.name}</h3>
                <div class="material-meta">
                    <span class="subject-tag" style="background-color: ${subject.color}">
                        ${subject.name}
                    </span>
                    <span class="material-size">
                        ${formatFileSize(material.file_size)}
                    </span>
                </div>
                <div class="material-actions">
                    <button class="btn btn-icon view-btn" data-id="${material._id}" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-icon download-btn" data-id="${material._id}" data-path="${material.file_path}" title="Download">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="btn btn-icon edit-btn" data-id="${material._id}" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-icon delete-btn" data-id="${material._id}" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Create list item
        const listItem = document.createElement('div');
        listItem.className = 'material-item';
        listItem.dataset.id = material._id;
        
        listItem.innerHTML = `
            <div class="material-icon ${material.file_type}">
                <i class="fas ${getFileIcon(material.file_type)}"></i>
            </div>
            <div class="material-content">
                <div class="material-header">
                    <h3>${material.name}</h3>
                    <div class="material-actions">
                        <button class="btn btn-icon view-btn" data-id="${material._id}" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-icon download-btn" data-id="${material._id}" data-path="${material.file_path}" title="Download">
                            <i class="fas fa-download"></i>
                        </button>
                        <button class="btn btn-icon edit-btn" data-id="${material._id}" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon delete-btn" data-id="${material._id}" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="material-description">
                    ${material.description || 'No description provided'}
                </div>
                <div class="material-meta">
                    <span class="subject-tag" style="background-color: ${subject.color}">
                        ${subject.name}
                    </span>
                    <span class="material-date">
                        <i class="fas fa-calendar-alt"></i>
                        ${formatDate(uploadDate)}
                    </span>
                    <span class="material-size">
                        <i class="fas fa-weight"></i>
                        ${formatFileSize(material.file_size)}
                    </span>
                    <span class="material-type">
                        <i class="fas fa-file"></i>
                        ${capitalizeFirstLetter(material.file_type)}
                    </span>
                </div>
            </div>
        `;
        
        // Add to containers
        materialsGrid.appendChild(gridItem);
        materialsList.appendChild(listItem);
    });
    
    // Use event delegation for better performance
    materialsGrid.addEventListener('click', handleMaterialAction);
    materialsList.addEventListener('click', handleMaterialAction);
}

// Handle material actions via event delegation
function handleMaterialAction(e) {
    // Find the closest button if we clicked on a child element
    const viewBtn = e.target.closest('.view-btn');
    const downloadBtn = e.target.closest('.download-btn');
    const editBtn = e.target.closest('.edit-btn');
    const deleteBtn = e.target.closest('.delete-btn');
    
    if (viewBtn) {
        e.preventDefault();
        e.stopPropagation();
        const materialId = viewBtn.getAttribute('data-id');
        if (materialId) {
            openMaterialDetailsModal(materialId);
        }
    } else if (downloadBtn) {
        e.preventDefault();
        e.stopPropagation();
        const materialId = downloadBtn.getAttribute('data-id');
        const filePath = downloadBtn.getAttribute('data-path');
        if (materialId) {
            downloadMaterial(materialId, filePath);
        }
    } else if (editBtn) {
        e.preventDefault();
        e.stopPropagation();
        const materialId = editBtn.getAttribute('data-id');
        if (materialId) {
            openEditMaterialModal(materialId);
        }
    } else if (deleteBtn) {
        e.preventDefault();
        e.stopPropagation();
        const materialId = deleteBtn.getAttribute('data-id');
        if (materialId) {
            openDeleteConfirmationModal(materialId);
        }
    }
}

// Update storage info
function updateStorageInfo(materials) {
    const totalStorage = 5 * 1024 * 1024; // 5MB
    const usedStorage = materials.reduce((total, material) => total + material.file_size, 0);
    
    document.getElementById('used-storage').textContent = formatFileSize(usedStorage);
    document.getElementById('total-storage').textContent = formatFileSize(totalStorage);
    
    // Add progress bar visualization
    const storageInfo = document.querySelector('.storage-info');
    if (storageInfo) {
        // Calculate percentage used
        const percentUsed = Math.min(100, (usedStorage / totalStorage) * 100);
        
        // Create or update progress bar
        let progressBar = document.querySelector('.storage-progress');
        if (!progressBar) {
            progressBar = document.createElement('div');
            progressBar.className = 'storage-progress';
            
            const progressFill = document.createElement('div');
            progressFill.className = 'storage-progress-fill';
            progressBar.appendChild(progressFill);
            
            storageInfo.appendChild(progressBar);
        }
        
        // Update progress fill
        const progressFill = progressBar.querySelector('.storage-progress-fill');
        progressFill.style.width = `${percentUsed}%`;
        
        // Add color based on usage
        if (percentUsed > 90) {
            progressFill.style.backgroundColor = '#e74c3c'; // Red for high usage
        } else if (percentUsed > 70) {
            progressFill.style.backgroundColor = '#f39c12'; // Orange for medium usage
        } else {
            progressFill.style.backgroundColor = '#2ecc71'; // Green for low usage
        }
    }
}

// Filter materials
function filterMaterials() {
    const searchTerm = document.getElementById('material-search').value.toLowerCase();
    const fileTypeFilter = document.getElementById('file-type-filter').value;
    const subjectFilter = document.getElementById('subject-filter').value;
    const sortFilter = document.getElementById('sort-filter').value;
    
    // Get all materials
    const materials = window.allMaterials || [];
    
    // Apply filters
    let filteredMaterials = materials.filter(material => {
        // Search term filter
        const matchesSearch = 
            material.name.toLowerCase().includes(searchTerm) ||
            (material.description && material.description.toLowerCase().includes(searchTerm));
        
        // File type filter
        const matchesFileType = fileTypeFilter === 'all' || material.file_type === fileTypeFilter;
        
        // Subject filter
        const matchesSubject = subjectFilter === 'all' || material.subject_id === subjectFilter;
        
        return matchesSearch && matchesFileType && matchesSubject;
    });
    
    // Apply sorting
    switch (sortFilter) {
        case 'newest':
            filteredMaterials.sort((a, b) => new Date(b.uploaded_at) - new Date(a.uploaded_at));
            break;
        case 'oldest':
            filteredMaterials.sort((a, b) => new Date(a.uploaded_at) - new Date(b.uploaded_at));
            break;
        case 'name_asc':
            filteredMaterials.sort((a, b) => a.name.localeCompare(b.name));
            break;
        case 'name_desc':
            filteredMaterials.sort((a, b) => b.name.localeCompare(a.name));
            break;
    }
    
    // Display filtered materials using the stored subjects map
    if (window.subjectsMap) {
        displayMaterials(filteredMaterials, window.subjectsMap);
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
            displayMaterials(filteredMaterials, subjectsMap);
        })
        .catch(error => {
            console.error('Error fetching subjects for filtering:', error);
        });
    }
}

// Open upload modal
function openUploadModal() {
    // Reset form
    const uploadForm = document.getElementById('upload-form');
    uploadForm.reset();
    
    // Reset file input
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Reset file preview
    document.getElementById('file-preview').style.display = 'none';
    document.getElementById('upload-area').style.display = 'flex';
    document.getElementById('upload-progress').style.display = 'none';
    
    // Reset progress bar
    const progressFill = document.getElementById('progress-fill');
    if (progressFill) {
        progressFill.style.width = '0%';
    }
    
    // Reset upload button
    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload';
    }
    
    // Show modal
    document.getElementById('upload-modal').classList.add('active');
    document.body.classList.add('modal-open');
}
// Open material details modal
function openMaterialDetailsModal(materialId) {
    if (!materialId) {
        console.error('No material ID provided for viewing');
        return;
    }
    
    const token = localStorage.getItem('accessToken');
    
    // Show loading in modal title
    document.getElementById('material-detail-title').textContent = 'Loading...';
    document.getElementById('material-details-modal').classList.add('active');
    document.body.classList.add('modal-open');
    
    // Fetch material details
    fetch(`https://study-o5hp.onrender.com/materials/${materialId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load material details');
        }
        return response.json();
    })
    .then(material => {
        // Use stored subjects map if available
        if (window.subjectsMap) {
            const subject = window.subjectsMap[material.subject_id] || { name: 'No Subject' };
            populateMaterialDetails(material, subject);
        } else {
            // Fetch subject info if needed
            return fetch('https://study-o5hp.onrender.com/subjects', {
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
                const subject = subjectsMap[material.subject_id] || { name: 'No Subject' };
                populateMaterialDetails(material, subject);
            });
        }
    })
    .catch(error => {
        console.error('Error fetching material details:', error);
        alert('Failed to load material details. Please try again.');
        closeModals();
    });
}

// Populate material details in modal
function populateMaterialDetails(material, subject) {
    // Populate modal
    document.getElementById('material-detail-title').textContent = material.name;
    document.getElementById('detail-name').textContent = material.name;
    document.getElementById('detail-description').textContent = material.description || 'No description provided';
    document.getElementById('detail-subject').textContent = subject.name;
    document.getElementById('detail-file-type').textContent = capitalizeFirstLetter(material.file_type);
    document.getElementById('detail-file-size').textContent = formatFileSize(material.file_size);
    document.getElementById('detail-uploaded-at').textContent = formatDate(new Date(material.uploaded_at));
    
    // Set appropriate icon
    const previewIcon = document.getElementById('material-preview');
    previewIcon.innerHTML = `<i class="fas ${getFileIcon(material.file_type)} material-preview-icon"></i>`;
    
    // Set up action buttons
    document.getElementById('download-btn').onclick = () => downloadMaterial(material._id, material.file_path);
    document.getElementById('edit-material-btn').onclick = () => {
        closeModals();
        openEditMaterialModal(material._id);
    };
    document.getElementById('delete-material-btn').onclick = () => {
        closeModals();
        openDeleteConfirmationModal(material._id);
    };
}

// Open edit material modal
function openEditMaterialModal(materialId) {
    if (!materialId) {
        console.error('No material ID provided for editing');
        return;
    }
    
    const token = localStorage.getItem('accessToken');
    
    // Show loading in the modal
    document.getElementById('edit-material-modal').classList.add('active');
    document.body.classList.add('modal-open');
    
    // Fetch material details
    fetch(`https://study-o5hp.onrender.com/materials/${materialId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load material details');
        }
        return response.json();
    })
    .then(material => {
        // Populate form
        document.getElementById('edit-material-id').value = material._id;
        document.getElementById('edit-name').value = material.name;
        document.getElementById('edit-description').value = material.description || '';
        document.getElementById('edit-subject-id').value = material.subject_id;
    })
    .catch(error => {
        console.error('Error fetching material details:', error);
        alert('Failed to load material details. Please try again.');
        closeModals();
    });
}

// Open delete confirmation modal
function openDeleteConfirmationModal(materialId) {
    if (!materialId) {
        console.error('No material ID provided for deletion');
        return;
    }
    
    // Store material ID for deletion
    document.getElementById('confirm-delete-btn').dataset.id = materialId;
    
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
    
    // Reset upload form if it exists
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.reset();
        
        // Reset file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Reset file preview
        const filePreview = document.getElementById('file-preview');
        const uploadArea = document.getElementById('upload-area');
        const uploadProgress = document.getElementById('upload-progress');
        
        if (filePreview) filePreview.style.display = 'none';
        if (uploadArea) uploadArea.style.display = 'flex';
        if (uploadProgress) uploadProgress.style.display = 'none';
    }
}

// Handle upload form submission
async function handleUploadFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const fileInput = document.getElementById('file-input');
    
    // Check if file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Check file size again
    if (file.size > 5 * 1024 * 1024) { // 5MB
        alert('File size exceeds the maximum limit of 5MB');
        return;
    }
    
    // Get form data
    const name = document.getElementById('name').value;
    const description = document.getElementById('description').value;
    const subjectId = document.getElementById('subject-id').value;
    
    // Create FormData object
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('description', description);
    formData.append('subject_id', subjectId);
    
    // Show loading state
    const uploadBtn = document.getElementById('upload-btn');
    const originalBtnText = uploadBtn.textContent;
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    
    // Show progress bar
    const progressBar = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    progressBar.style.display = 'block';
    
    try {
        // Create a mock progress simulation (since we don't have actual upload progress)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 90) {
                clearInterval(progressInterval);
            }
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Uploading... ${progress}%`;
        }, 200);
        
        // Upload file
        const response = await fetch('https://study-o5hp.onrender.com/materials', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });
        
        // Clear progress interval
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to upload material');
        }
        
        // Get the uploaded material data
        const uploadedMaterial = await response.json();
        
        // Update our local data
        if (window.allMaterials) {
            window.allMaterials.push(uploadedMaterial);
        }
        
        // Complete progress
        progressFill.style.width = '100%';
        progressText.textContent = 'Upload complete!';
        
        // Close modal and reload materials after a short delay
        setTimeout(() => {
            closeModals();
            
            // Update UI without full reload
            if (window.subjectsMap) {
                displayMaterials(window.allMaterials, window.subjectsMap);
                updateStorageInfo(window.allMaterials);
            } else {
                loadMaterials();
            }
            
            // IMPORTANT: Reset the form and file input properly
            document.getElementById('upload-form').reset();
            fileInput.value = '';
            
            // Reset file preview
            document.getElementById('file-preview').style.display = 'none';
            document.getElementById('upload-area').style.display = 'flex';
            document.getElementById('upload-progress').style.display = 'none';
            
            // Re-enable the upload button
            uploadBtn.disabled = false;
            uploadBtn.textContent = originalBtnText;
            
        }, 1000);
        
    } catch (error) {
        console.error('Error uploading material:', error);
        alert(`Error: ${error.message}`);
        
        // Reset button
        uploadBtn.disabled = false;
        uploadBtn.textContent = originalBtnText;
        
        // Hide progress bar
        document.getElementById('upload-progress').style.display = 'none';
    }
}

// Handle edit material form submission
async function handleEditMaterialFormSubmit(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('accessToken');
    const materialId = document.getElementById('edit-material-id').value;
    
    if (!materialId) {
        console.error('No material ID found for editing');
        alert('Error: Could not identify which material to edit.');
        closeModals();
        return;
    }
    
    // Get form data
    const formData = {
        name: document.getElementById('edit-name').value,
        description: document.getElementById('edit-description').value,
        subject_id: document.getElementById('edit-subject-id').value
    };
    
    // Show loading state
    const saveBtn = document.getElementById('save-material-btn');
    const originalBtnText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/materials/${materialId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to update material');
        }
        
        // Get the updated material data
        const updatedMaterial = await response.json();
        
        // Update our local data
        if (window.allMaterials) {
            window.allMaterials = window.allMaterials.map(m => 
                m._id === materialId ? updatedMaterial : m
            );
        }
        
        // Close modal
        closeModals();
        
        // Update UI without full reload
        if (window.subjectsMap) {
            displayMaterials(window.allMaterials, window.subjectsMap);
        } else {
            loadMaterials();
        }
        
    } catch (error) {
        console.error('Error updating material:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        saveBtn.disabled = false;
        saveBtn.textContent = originalBtnText;
    }
}

// Delete material
async function deleteMaterial() {
    const token = localStorage.getItem('accessToken');
    const materialId = document.getElementById('confirm-delete-btn').dataset.id;
    
    if (!materialId) {
        console.error('No material ID found for deletion');
        alert('Error: Could not identify which material to delete.');
        closeModals();
        return;
    }
    
    // Show loading state
    const deleteBtn = document.getElementById('confirm-delete-btn');
    const originalBtnText = deleteBtn.textContent;
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
    
    try {
        const response = await fetch(`https://study-o5hp.onrender.com/materials/${materialId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to delete material');
        }
        
        // Update our local data
        if (window.allMaterials) {
            window.allMaterials = window.allMaterials.filter(m => m._id !== materialId);
        }
        
        // Close modal
        closeModals();
        
        // Update UI without full reload
        if (window.subjectsMap) {
            displayMaterials(window.allMaterials, window.subjectsMap);
            updateStorageInfo(window.allMaterials);
        } else {
            loadMaterials();
        }
        
    } catch (error) {
        console.error('Error deleting material:', error);
        alert(`Error: ${error.message}`);
    } finally {
        // Always reset button state
        deleteBtn.disabled = false;
        deleteBtn.textContent = originalBtnText;
        
        // Clear the material ID
        deleteBtn.dataset.id = '';
    }
}

// Download material
async function downloadMaterial(materialId, filePath) {
    if (!materialId) {
        console.error('No material ID provided for download');
        return;
    }
    
    const token = localStorage.getItem('accessToken');
    
    try {
        // Add download notification
        const notification = document.createElement('div');
        notification.className = 'download-notification';
        
        // Find the material in our local data
        let material;
        if (window.allMaterials) {
            material = window.allMaterials.find(m => m._id === materialId);
        }
        
        // Update notification with material info if available
        if (material) {
            notification.innerHTML = `
                <i class="fas ${getFileIcon(material.file_type)}"></i>
                <span>Downloading ${material.name}...</span>
            `;
        } else {
            notification.innerHTML = `
                <i class="fas fa-download"></i>
                <span>Downloading file...</span>
            `;
        }
        
        document.body.appendChild(notification);
        
        // Show notification
        setTimeout(() => {
            notification.classList.add('active');
        }, 10);

        // Create download URL
        const downloadUrl = `https://study-o5hp.onrender.com/materials/download/${materialId}?token=${token}`;
        
        // Create a temporary anchor element to trigger the download
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.target = '_blank'; // Open in new tab to handle potential auth issues
        a.download = material ? material.name : 'download';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // Hide notification after a few seconds
        setTimeout(() => {
            notification.classList.remove('active');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
        
    } catch (error) {
        console.error('Error downloading material:', error);
        alert(`Error: ${error.message}`);
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
        document.getElementById('materials-grid').classList.add('view-active');
        document.getElementById('materials-list').classList.remove('view-active');
    } else {
        document.getElementById('materials-grid').classList.remove('view-active');
        document.getElementById('materials-list').classList.add('view-active');
    }
}

// Helper functions
function formatDate(date) {
    const options = { month: 'short', day: 'numeric', year: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + ' B';
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + ' KB';
    } else {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
}

function getFileIcon(type) {
    switch (type) {
        case 'document':
            return 'fa-file-word';
        case 'image':
            return 'fa-file-image';
        case 'spreadsheet':
            return 'fa-file-excel';
        case 'pdf':
            return 'fa-file-pdf';
        default:
            return 'fa-file';
    }
}

function capitalizeFirstLetter(string) {
    if (!string) return '';
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
