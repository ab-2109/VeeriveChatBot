// Variables to track state
let waitingForClarification = false;
let activeSessionId = null;

const API_BASE = 'https://3.86.52.25:8000'; // At minimum, add https://

// Core functions for the chat interface
document.addEventListener('DOMContentLoaded', () => {
    console.log("Chat.js loaded!");
    
    // Get DOM elements
    const messagesContainer = document.getElementById('messages-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    const typingIndicator = document.getElementById('typing-indicator');
    
    // Debug log elements
    console.log("Elements found:", {
        messagesContainer: !!messagesContainer,
        questionInput: !!questionInput,
        sendButton: !!sendButton,
        typingIndicator: !!typingIndicator
    });
    
    // Attach event listener to send button
    sendButton.addEventListener('click', () => {
        console.log("Send button clicked, clarification mode:", waitingForClarification);
        if (waitingForClarification) {
            submitClarification();
        } else {
            sendQuestion();
        }
    });
    
    // Auto resize the textarea as user types
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        
        questionInput.style.height = questionInput.scrollHeight + 'px';
    });
    
    // Send message on Enter (but allow Shift+Enter for new lines)
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (waitingForClarification) {
                submitClarification();
            } else {
                sendQuestion();
            }
        }
    });
});

// Send a new question
function sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question) return;

    addMessage(question, 'user');
    input.value = '';
    disableInput();

    fetch(`${API_BASE}/graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question })
    })
    .then(res => res.json())
    .then(data => {
        handleApiResponse(data);
    })
    .catch(err => {
        console.error("Graph error:", err);
        addMessage(`Error: ${err.message}`, 'system');
        resetConversationState();
    });
}


// Submit a clarification answer
function submitClarification() {
    const input = document.getElementById('question-input');
    const answer = input.value.trim();
    if (!answer || !activeSessionId) return;

    addMessage(answer, 'user');
    input.value = '';
    input.placeholder = 'Processing...';
    disableInput();

    fetch(`${API_BASE}/clarify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: activeSessionId,
            answer: answer
        })
    })
    .then(res => res.json())
    .then(data => {
        handleApiResponse(data);  // ✅ reuse the same response handler
    })
    .catch(err => {
        console.error("Clarification error:", err);
        addMessage(`Error: ${err.message}`, 'system');
        resetConversationState();
    });
}


// Fetch response from the API
function fetchGraphResponse(question, metadata = null) {
    debugDisplay("Sending request to backend: " + question);
    
    // Use relative URL instead of hardcoded localhost
    fetch(`${API_BASE}/graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            query: question,
            metadata: metadata 
        })
    })
    .then(response => {
        hideTypingIndicator();
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    })
    .then(data => {
        debugDisplay("API response received: " + JSON.stringify(data).substring(0, 100) + "...");
        handleApiResponse(data);
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'system');
        resetConversationState();
    });
}

// Handle the API response
function handleApiResponse(data) {
    console.log("Response status:", data.status);
    console.log("Full data:", data);
    
    if (data.status === "clarification_needed") {
        console.log("Setting up for clarification, session ID:", data.session_id);
        
        activeSessionId = data.session_id;
        waitingForClarification = true;
        
        const question = data.clarification_question || "Please provide additional information";
        addMessage(question, 'system');
        
        enableInput();
        document.getElementById('question-input').focus();
        document.getElementById('send-button').textContent = "Submit";
        document.getElementById('question-input').placeholder = "Provide additional information...";
    } 
    else if (data.status === "complete") {
        waitingForClarification = false;
        activeSessionId = null;

        document.getElementById('send-button').textContent = "Send";
        document.getElementById('question-input').placeholder = "Ask a finance question...";

        const formattedResponse = formatResponse(
            data.result?.generated_response || data.result
        );
        addMessage(formattedResponse, 'assistant');

        // ✅ Re-enable input for next query
        enableInput();
        document.getElementById('question-input').focus();
    }
    else if (data.status === "error") {
        const errorMsg = data.error || "Unknown error occurred";
        console.error("API error:", errorMsg);
        addMessage(`Error: ${errorMsg}`, 'system');
        resetConversationState();
    }
    else {
        console.error("Unknown API status:", data.status);
        addMessage(`Error: Received unknown status '${data.status}' from server`, 'system');
        resetConversationState();
    }
}


// Reset conversation state
function resetConversationState() {
    activeSessionId = null;
    waitingForClarification = false;

    enableInput();
    document.getElementById('send-button').textContent = "Send";
    document.getElementById('question-input').placeholder = "Ask a finance question...";
    document.getElementById('question-input').value = '';
}

// Add a message to the chat
function addMessage(content, sender) {
    debugDisplay("Adding " + sender + " message");
    const messagesContainer = document.getElementById('messages-container');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (typeof content === 'string') {
        messageContent.innerHTML = content;
    } else {
        messageContent.innerHTML = JSON.stringify(content, null, 2);
    }
    
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Format the response
function formatResponse(result) {
    console.log("Formatting result:", result);
    
    if (!result) return "No result received";
    
    // Handle undefined result
    if (typeof result === 'undefined') {
        return "Error: Received undefined response from server";
    }
    
    // Check if we have the nested generated_response structure
    if (result.generated_response) {
        console.log("Found generated_response, using that");
        return formatResponse(result.generated_response);
    }
    
    // Check if we have the structured/conversational format
    if (result.structured && result.conversational) {
        console.log("Using structured/conversational format");
        
        const structuredData = result.structured.data;
        const conversationalData = result.conversational.data;
        
        if (!structuredData) {
            return markdownToHtml(conversationalData) || "No data received";
        }
        
        // Create tabs UI
        let html = `
        <div class="response-tabs">
            <button class="tab-btn active" onclick="switchTab(event, 'structured')">Structured</button>
            <button class="tab-btn" onclick="switchTab(event, 'conversational')">Conversational</button>
        </div>
        `;
        
        // Structured tab content
        html += `<div id="structured-tab" class="tab-content active">`;
        
        // Module 1
        if (structuredData.module1) {
            html += `
            <div class="dropdown">
                <div class="dropdown-header" onclick="toggleDropdown(this)">
                    <h3>Key Concepts and Models</h3>
                    <span class="dropdown-arrow">▼</span>
                </div>
                <div class="dropdown-content">
                    <p>${structuredData.module1.overview || ""}</p>
            `;
            
            // Model details
            if (structuredData.module1.model_details?.length) {
                html += `<h4>Model Details</h4><ul>`;
                structuredData.module1.model_details.forEach(detail => {
                    html += `<li>${detail}</li>`;
                });
                html += `</ul>`;
            }
            
            // Model comparison
            if (structuredData.module1.model_comparison?.headers) {
                html += `<h4>Model Comparison</h4>
                    <div class="table-wrapper">
                        <table class="response-table">
                            <thead>
                                <tr>`;
                
                structuredData.module1.model_comparison.headers.forEach(header => {
                    html += `<th>${header}</th>`;
                });
                
                html += `</tr>
                            </thead>
                            <tbody>`;
                
                if (structuredData.module1.model_comparison.rows) {
                    structuredData.module1.model_comparison.rows.forEach(row => {
                        html += `<tr>`;
                        row.forEach(cell => {
                            html += `<td>${cell}</td>`;
                        });
                        html += `</tr>`;
                    });
                }
                
                html += `</tbody>
                        </table>
                    </div>`;
            }
            
            // Examples
            if (structuredData.module1.examples?.length) {
                html += `<h4>Examples</h4><ul>`;
                structuredData.module1.examples.forEach(example => {
                    html += `<li>${example}</li>`;
                });
                html += `</ul>`;
            }
            
            html += `</div></div>`;
        }
        
        // Module 2
        if (structuredData.module2) {
            html += `
            <div class="dropdown">
                <div class="dropdown-header" onclick="toggleDropdown(this)">
                    <h3>Strategic Shifts and Innovation</h3>
                    <span class="dropdown-arrow">▼</span>
                </div>
                <div class="dropdown-content">`;
            
            if (structuredData.module2.major_events?.length) {
                html += `<h4>Major Events</h4><ul>`;
                structuredData.module2.major_events.forEach(event => {
                    html += `<li>${event}</li>`;
                });
                html += `</ul>`;
            }
            
            if (structuredData.module2.expert_opinions?.length) {
                html += `<h4>Expert Opinions</h4><ul>`;
                structuredData.module2.expert_opinions.forEach(opinion => {
                    html += `<li>${opinion}</li>`;
                });
                html += `</ul>`;
            }
            
            html += `</div></div>`;
        }
        
        // Module 3
        if (structuredData.module3) {
            html += `
            <div class="dropdown">
                <div class="dropdown-header" onclick="toggleDropdown(this)">
                    <h3>Trend Analysis</h3>
                    <span class="dropdown-arrow">▼</span>
                </div>
                <div class="dropdown-content">`;
            
            if (structuredData.module3.key_trends?.length) {
                html += `<h4>Key Trends</h4><ul>`;
                structuredData.module3.key_trends.forEach(trend => {
                    html += `<li>${trend}</li>`;
                });
                html += `</ul>`;
            }
            
            if (structuredData.module3.associated_themes?.length) {
                html += `<h4>Associated Themes</h4><ul>`;
                structuredData.module3.associated_themes.forEach(theme => {
                    html += `<li>${theme}</li>`;
                });
                html += `</ul>`;
            }
            
            html += `</div></div>`;
        }
        
        // Module 4
        if (structuredData.module4) {
            html += `
            <div class="dropdown">
                <div class="dropdown-header" onclick="toggleDropdown(this)">
                    <h3>Global Context</h3>
                    <span class="dropdown-arrow">▼</span>
                </div>
                <div class="dropdown-content">`;
            
            if (structuredData.module4.global_events?.length) {
                html += `<h4>Global Events</h4><ul>`;
                structuredData.module4.global_events.forEach(event => {
                    html += `<li>${event}</li>`;
                });
                html += `</ul>`;
            }
            
            if (structuredData.module4.global_model_shifts?.length) {
                html += `<h4>Global Model Shifts</h4><ul>`;
                structuredData.module4.global_model_shifts.forEach(shift => {
                    html += `<li>${shift}</li>`;
                });
                html += `</ul>`;
            }
            
            html += `</div></div>`;
        }
        
        html += `</div>`;
        
        // Conversational tab content
        html += `<div id="conversational-tab" class="tab-content">`;
        html += markdownToHtml(conversationalData) || "<p>No conversational response available</p>";
        html += `</div>`;
        
        // Add necessary styles
        html += `
        <style>
            .response-tabs {
                display: flex;
                margin-bottom: 1rem;
                border-bottom: 1px solid #ddd;
            }
            
            .tab-btn {
                padding: 0.5rem 1rem;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1rem;
                border-bottom: 2px solid transparent;
            }
            
            .tab-btn.active {
                border-bottom: 2px solid #0070f3;
                font-weight: bold;
            }
            
            .tab-content {
                display: none;
                padding: 0.5rem;
            }
            
            .tab-content.active {
                display: block;
            }
            
            #conversational-tab h1 {
                font-size: 1.6rem;
                margin-top: 1.2rem;
                margin-bottom: 0.8rem;
                color: #0070f3;
            }
            
            #conversational-tab h2 {
                font-size: 1.4rem;
                margin-top: 1.1rem;
                margin-bottom: 0.7rem;
                color: #0076f5;
            }
            
            #conversational-tab h3 {
                font-size: 1.3rem;
                margin-top: 1rem;
                margin-bottom: 0.6rem;
                color: #0080f7;
            }
            
            #conversational-tab h4 {
                font-size: 1.2rem;
                margin-top: 0.9rem;
                margin-bottom: 0.5rem;
                color: #0088f9;
            }
            
            #conversational-tab h5 {
                font-size: 1.1rem;
                margin-top: 0.8rem;
                margin-bottom: 0.4rem;
                color: #0092fb;
            }
            
            #conversational-tab ul {
                margin-left: 1.5rem;
                margin-bottom: 1rem;
            }
            
            #conversational-tab li {
                margin-bottom: 0.3rem;
            }
            
            #conversational-tab p {
                margin-bottom: 1rem;
            }
            
            .dropdown {
                margin-bottom: 1rem;
                border: 1px solid #ddd;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .dropdown-header {
                padding: 0.75rem;
                cursor: pointer;
                background-color: #f9f9f9;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .dropdown-header h3 {
                margin: 0;
                font-size: 1.1rem;
                color: #0070f3;
            }
            
            .dropdown-arrow {
                font-size: 0.8rem;
            }
            
            .dropdown-content {
                padding: 0;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
            }
            
            .table-wrapper {
                overflow-x: auto;
                margin: 1rem 0;
            }
            
            .response-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .response-table th, .response-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            
            .response-table th {
                background-color: #f2f2f2;
            }
        </style>
        `;
        
        return html;
    }
    
    // Fallback to displaying JSON for other formats
    try {
        if (typeof result === 'string') {
            return result;
        } else {
            return `<pre class="json-response">${JSON.stringify(result, null, 2)}</pre>`;
        }
    } catch (e) {
        return "Error formatting response: " + e.message;
    }
}

// Markdown to HTML converter
function markdownToHtml(markdown) {
    if (!markdown) return "";
    
    // Process headers (#### Heading -> <h4>Heading</h4>)
    const withHeaders = markdown
        .replace(/##### (.*?)(\n|$)/g, '<h5>$1</h5>')
        .replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>')
        .replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>')
        .replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>')
        .replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>');
    
    // Process lists (- Item -> <li>Item</li>)
    let inList = false;
    let withLists = "";
    withHeaders.split('\n').forEach(line => {
        if (line.trim().match(/^- /)) {
            if (!inList) {
                withLists += "<ul>";
                inList = true;
            }
            withLists += `<li>${line.trim().substring(2)}</li>`;
        } else {
            if (inList) {
                withLists += "</ul>";
                inList = false;
            }
            withLists += line + "\n";
        }
    });
    if (inList) withLists += "</ul>";
    
    // Process paragraphs (empty line between paragraphs -> <p>paragraph</p>)
    const withParagraphs = withLists
        .replace(/\n\n/g, '</p><p>')
        .replace(/^\s*(.+)/, '<p>$1');
    
    // If it doesn't end with </p>, add it
    const trimmed = withParagraphs.trim();
    if (!trimmed.endsWith('</p>') && !trimmed.endsWith('</ul>')) {
        return trimmed + '</p>';
    }
    
    return trimmed;
}

// Helper functions
function disableInput() {
    document.getElementById('question-input').disabled = true;
    document.getElementById('send-button').disabled = true;
}


function enableInput() {
    document.getElementById('question-input').disabled = false;
    document.getElementById('send-button').disabled = false;
}

function showTypingIndicator() {
    document.getElementById('typing-indicator').classList.remove('hidden');
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator').classList.add('hidden');
}

// Global functions for dropdown and tab switching
function switchTab(event, tabName) {
    console.log("Switching to tab:", tabName);
    
    // Hide all tab content
    const tabContents = document.querySelectorAll('.tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    
    // Show the selected tab content and mark the button as active
    const targetTab = document.getElementById(tabName + '-tab');
    if (targetTab) {
        targetTab.classList.add('active');
        event.currentTarget.classList.add('active');
    } else {
        console.error("Tab not found:", tabName + '-tab');
    }
}

function toggleDropdown(header) {
    console.log("Toggling dropdown");
    
    const content = header.nextElementSibling;
    const arrow = header.querySelector('.dropdown-arrow');
    
    if (content.style.maxHeight) {
        content.style.maxHeight = null;
        arrow.textContent = '▼';
    } else {
        content.style.maxHeight = content.scrollHeight + "px";
        arrow.textContent = '▲';
    }
}