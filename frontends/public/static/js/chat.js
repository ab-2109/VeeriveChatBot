// Variables to track state
let waitingForClarification = false;
let activeSessionId = null;

const API_BASE = '/api'; 
const DEBUG = true;

// Debug function
function debug(label, data) {
    if (!DEBUG) return;
    console.log(`DEBUG - ${label}:`, data);
    
    // Also show in UI if we have a debug panel
    const debugPanel = document.getElementById('debug-panel');
    if (debugPanel) {
        const entry = document.createElement('div');
        entry.innerHTML = `<strong>${label}</strong>: ${JSON.stringify(data, null, 2)}`;
        debugPanel.appendChild(entry);
        debugPanel.scrollTop = debugPanel.scrollHeight;
    }
}

// Core functions for the chat interface
document.addEventListener('DOMContentLoaded', () => {
    // Create debug panel
    const debugPanel = document.createElement('div');
    debugPanel.id = 'debug-panel';
    debugPanel.style = 'position: fixed; bottom: 10px; right: 10px; width: 300px; height: 200px; background: #f5f5f5; border: 1px solid #ccc; overflow: auto; padding: 10px; font-size: 11px; z-index: 9999;';
    document.body.appendChild(debugPanel);
    
    // Get DOM elements
    const messagesContainer = document.getElementById('messages-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    
    // Test API connection
    fetch(`${API_BASE}/`)
        .then(res => res.json())
        .then(data => {
            debug('API Connection Success', data);
        })
        .catch(err => {
            debug('API Connection Error', err.message);
        });
    
    // Attach event listener to send button
    sendButton.addEventListener('click', () => {
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

    debug('Sending question', { query: question });
    
    fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            query: question,
            metadata: {}
        })
    })
    .then(res => {
        debug('Response status', { status: res.status });
        return res.json();
    })
    .then(data => {
        debug('Response data', data);
        processApiResponse(data);
    })
    .catch(err => {
        debug('Error', err);
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
    
    debug('Submitting clarification', { 
        session_id: activeSessionId,
        answer: answer
    });

    fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: answer,
            session_id: activeSessionId
        })
    })
    .then(res => {
        debug('Clarification response status', { status: res.status });
        return res.json();
    })
    .then(data => {
        debug('Clarification response data', data);
        processApiResponse(data);
    })
    .catch(err => {
        debug('Clarification error', err);
        addMessage(`Error: ${err.message}`, 'system');
        resetConversationState();
    });
}

// Process API response with comprehensive checks for all possible formats
function processApiResponse(data) {
    debug('Processing API response', data);
    
    // Check for clarification needed status or questions
    if (data.status === "clarification_needed" || data.clarification_question) {
        handleClarificationNeeded(data);
    } 
    else if (data.status === "complete") {
        handleCompletedResponse(data);
    }
    else if (data.status === "error") {
        const errorMsg = data.error || "Unknown error occurred";
        addMessage(`Error: ${errorMsg}`, 'system');
        resetConversationState();
    }
    else {
        addMessage(`Error: Received unknown response from server`, 'system');
        resetConversationState();
    }
}

// Handle clarification needed
function handleClarificationNeeded(data) {
    activeSessionId = data.session_id;
    waitingForClarification = true;
    
    // Find clarification question - prioritize direct field
    const question = data.clarification_question || 
                     (data.result && data.result.clarification_question) ||
                     "Please provide additional information";
    
    debug('Found clarification question', { question });
    
    // Display the question to the user
    addMessage(question, 'system');
    
    // Update UI to show we're awaiting clarification
    document.getElementById('send-button').textContent = "Submit";
    document.getElementById('question-input').placeholder = "Provide additional information...";
    enableInput();
    document.getElementById('question-input').focus();
}

// Handle completed response
// Handle completed response
function handleCompletedResponse(data) {
    waitingForClarification = false;
    activeSessionId = null;

    // Reset UI elements
    document.getElementById('send-button').textContent = "Send";
    document.getElementById('question-input').placeholder = "Ask a finance question...";
    
    // Log the entire response to debug panel
    debug("FULL RESPONSE", data);
    
    try {
        // Get the result object - it should be in data.result
        let result = data.result;
        
        debug("EXTRACTED RESULT", result);
        
        // Create a container for the entire message
        let messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        let hasStructuredData = false;
        let hasConversationalData = false;
        
        // FIRST: Add structured data if available
        if (result && result.structured && result.structured.data) {
            debug("ADDING STRUCTURED DATA", result.structured.data);
            hasStructuredData = true;
            
            // Create structured section div
            let structuredDiv = document.createElement('div');
            structuredDiv.className = 'structured-response';
            
            // Add heading
            let heading = document.createElement('h3');
            heading.textContent = 'Detailed Analysis';
            structuredDiv.appendChild(heading);
            
            // Process structured data
            const structuredData = result.structured.data;
            
            // Add module1 data if exists
            if (structuredData.module1) {
                let module = structuredData.module1;
                
                // Overview section
                if (module.overview) {
                    let section = document.createElement('div');
                    section.className = 'section';
                    section.innerHTML = `<h4>Overview</h4><p>${module.overview}</p>`;
                    structuredDiv.appendChild(section);
                }
                
                // Model details as dropdown
                if (module.model_details && module.model_details.length) {
                    let details = document.createElement('details');
                    details.className = 'dropdown-section';
                    details.open = true;
                    
                    let summary = document.createElement('summary');
                    summary.innerHTML = '<h4>Key Points</h4>';
                    details.appendChild(summary);
                    
                    let ul = document.createElement('ul');
                    module.model_details.forEach(detail => {
                        let li = document.createElement('li');
                        li.textContent = detail;
                        ul.appendChild(li);
                    });
                    
                    details.appendChild(ul);
                    structuredDiv.appendChild(details);
                }
            }
            
            // Process other modules if they exist
            Object.keys(structuredData).forEach(key => {
                if (key !== 'module1' && structuredData[key]) {
                    let moduleData = structuredData[key];
                    let details = document.createElement('details');
                    details.className = 'dropdown-section';
                    
                    let summary = document.createElement('summary');
                    summary.innerHTML = `<h4>${key.replace('module', 'Module ')}</h4>`;
                    details.appendChild(summary);
                    
                    let content = document.createElement('div');
                    content.style.padding = '15px';
                    
                    if (typeof moduleData === 'object') {
                        // Handle object data
                        Object.keys(moduleData).forEach(subKey => {
                            let subSection = document.createElement('div');
                            subSection.innerHTML = `<strong>${subKey}:</strong> ${JSON.stringify(moduleData[subKey], null, 2)}`;
                            content.appendChild(subSection);
                        });
                    } else {
                        content.textContent = moduleData;
                    }
                    
                    details.appendChild(content);
                    structuredDiv.appendChild(details);
                }
            });
            
            // Add structured div to message content
            messageContent.appendChild(structuredDiv);
        }
        
        // SECOND: Add conversational content if available
        if (result && result.conversational && result.conversational.data) {
            debug("ADDING CONVERSATIONAL DATA", result.conversational.data);
            hasConversationalData = true;
            
            // Add separator if we have both sections
            if (hasStructuredData) {
                let separator = document.createElement('hr');
                separator.className = 'response-separator';
                messageContent.appendChild(separator);
            }
            
            // Create conversational section
            let conversationalDiv = document.createElement('div');
            conversationalDiv.className = 'conversational-part';
            
            // Format conversational content
            const conversationalContent = typeof result.conversational.data === 'string' 
                ? markdownToHtml(result.conversational.data)
                : JSON.stringify(result.conversational.data, null, 2);
            
            conversationalDiv.innerHTML = conversationalContent;
            messageContent.appendChild(conversationalDiv);
        }
        
        // Fallback: If no structured or conversational data found, show what we have
        if (!hasStructuredData && !hasConversationalData) {
            debug("NO STRUCTURED/CONVERSATIONAL DATA FOUND, SHOWING FALLBACK");
            
            let fallbackDiv = document.createElement('div');
            fallbackDiv.className = 'conversational-part';
            
            // Try to extract any text content
            let content = '';
            if (result && typeof result === 'string') {
                content = result;
            } else if (result && result.response) {
                content = result.response;
            } else if (result && result.text) {
                content = result.text;
            } else {
                content = "I generated a response, but couldn't format it properly. Please try again.";
            }
            
            fallbackDiv.innerHTML = markdownToHtml(content);
            messageContent.appendChild(fallbackDiv);
        }
        
        // Create the message div with combined class
        let messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant combined';
        messageDiv.appendChild(messageContent);
        
        // Add to messages container
        document.getElementById('messages-container').appendChild(messageDiv);
        
        // Auto-scroll to the bottom
        const messagesContainer = document.getElementById('messages-container');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
    } catch (err) {
        debug("ERROR IN RESPONSE PROCESSING", err.message);
        // Fallback to a simple message display
        addMessage("Sorry, there was an error processing the response. Please try again.", 'system');
    }
    
    enableInput();
    document.getElementById('question-input').focus();
}
// Add this helper function for markdown conversion
function markdownToHtml(markdown) {
    if (!markdown) return "";
    
    // Replace headers
    let html = markdown
        .replace(/##### (.*?)(\n|$)/g, '<h5>$1</h5>')
        .replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>')
        .replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>')
        .replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>')
        .replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>');
    
    // Replace lists
    html = html.replace(/^\s*[*-] (.*?)(\n|$)/gm, '<li>$1</li>');
    html = html.replace(/<li>(.*?)<\/li>(\n<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    // Replace paragraphs
    html = html.replace(/^(?!<[uh]|<li)(.*?)(\n|$)/gm, '<p>$1</p>');
    
    return html;
}

// Add additional debug helper
function debugPrintObject(obj, label) {
    try {
        debugDisplay(`${label}: ${JSON.stringify(obj, null, 2).substring(0, 500)}...`);
    } catch (e) {
        debugDisplay(`${label}: [Error stringifying object: ${e.message}]`);
    }
}

// Update the addMessage function to support the "structured" class
function addMessage(content, sender) {
    const messagesContainer = document.getElementById('messages-container');
    
    const messageDiv = document.createElement('div');
    
    // Check if sender has additional classes (like "assistant structured")
    const senderClasses = sender.split(' ');
    const primarySender = senderClasses[0]; // "assistant" in "assistant structured"
    
    // Set the base class
    messageDiv.className = `message ${primarySender}`;
    
    // Add any additional classes
    if (senderClasses.length > 1) {
        for (let i = 1; i < senderClasses.length; i++) {
            messageDiv.classList.add(senderClasses[i]);
        }
    }
    
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

// Helper functions
function disableInput() {
    document.getElementById('question-input').disabled = true;
    document.getElementById('send-button').disabled = true;
}

function enableInput() {
    document.getElementById('question-input').disabled = false;
    document.getElementById('send-button').disabled = false;
}