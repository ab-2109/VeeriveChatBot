// Variables to track state
let waitingForClarification = false;
let activeSessionId = null;

const API_BASE = 'https://54.205.162.22.nip.io'; 
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
    document.getElementById('send-button').textContent = "Send";
    document.getElementById('question-input').placeholder = "Ask a finance question...";
    debug("FULL RESPONSE", data);
    try {
        let result = data.result;
        // If result is a JSON string, parse it
        if (typeof result === 'string') {
            try {
                result = JSON.parse(result);
            } catch (e) {
                // Not JSON, fallback to string
            }
        }
        debug("PARSED RESULT", result);
        let messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        let hasStructuredData = false;
        let hasConversationalData = false;
        let hasPDFData = false;
        // --- Structured Data ---
        if (result && result.structured && result.structured.data) {
            hasStructuredData = true;
            let structuredDiv = document.createElement('div');
            structuredDiv.className = 'structured-response';
            let heading = document.createElement('h3');
            heading.textContent = 'Detailed Analysis';
            structuredDiv.appendChild(heading);
            const structuredData = result.structured.data;
            // Render tables if present
            if (structuredData.table) {
                structuredDiv.innerHTML += renderTable(structuredData.table);
            }
            // Render other modules as before
            Object.keys(structuredData).forEach(key => {
                if (key !== 'table' && structuredData[key]) {
                    let section = document.createElement('div');
                    section.className = 'section';
                    section.innerHTML = `<strong>${key}:</strong> ${JSON.stringify(structuredData[key], null, 2)}`;
                    structuredDiv.appendChild(section);
                }
            });
            messageContent.appendChild(structuredDiv);
        }
        // --- PDF Content ---
        if (result && result.pdf_content) {
            hasPDFData = true;
            let pdfDiv = document.createElement('div');
            pdfDiv.className = 'pdf-content-block';
            let heading = document.createElement('h3');
            heading.textContent = 'PDF Extracts';
            pdfDiv.appendChild(heading);
            // pdf_content can be array or object
            let pdfSections = Array.isArray(result.pdf_content) ? result.pdf_content : [result.pdf_content];
            pdfSections.forEach(section => {
                pdfDiv.innerHTML += renderPDFSection(section);
            });
            messageContent.appendChild(pdfDiv);
        }
        // --- Conversational Data ---
        if (result && result.conversational && result.conversational.data) {
            hasConversationalData = true;
            if (hasStructuredData || hasPDFData) {
                let separator = document.createElement('hr');
                separator.className = 'response-separator';
                messageContent.appendChild(separator);
            }
            let conversationalDiv = document.createElement('div');
            conversationalDiv.className = 'conversational-part';
            const conversationalContent = typeof result.conversational.data === 'string' 
                ? markdownToHtml(result.conversational.data)
                : JSON.stringify(result.conversational.data, null, 2);
            conversationalDiv.innerHTML = conversationalContent;
            messageContent.appendChild(conversationalDiv);
        }
        // --- Fallback ---
        if (!hasStructuredData && !hasConversationalData && !hasPDFData) {
            let fallbackDiv = document.createElement('div');
            fallbackDiv.className = 'conversational-part';
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
        let messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant combined';
        messageDiv.appendChild(messageContent);
        document.getElementById('messages-container').appendChild(messageDiv);
        const messagesContainer = document.getElementById('messages-container');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    } catch (err) {
        debug("ERROR IN RESPONSE PROCESSING", err.message);
        addMessage("Sorry, there was an error processing the response. Please try again.", 'system');
    }
    enableInput();
    document.getElementById('question-input').focus();
}

// --- Helper: Render Table ---
function renderTable(tableData) {
    if (!Array.isArray(tableData) || tableData.length === 0) return '';
    let html = '<table class="response-table"><thead><tr>';
    // Table headers from keys of first row
    Object.keys(tableData[0]).forEach(key => {
        html += `<th>${key}</th>`;
    });
    html += '</tr></thead><tbody>';
    tableData.forEach(row => {
        html += '<tr>';
        Object.values(row).forEach(val => {
            html += `<td>${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

// --- Helper: Render PDF Section ---
function renderPDFSection(pdfSection) {
    let html = '<div class="pdf-section">';
    if (pdfSection.title) {
        html += `<h4 class="pdf-title">${pdfSection.title}</h4>`;
    }
    if (pdfSection.page) {
        html += `<div class="pdf-page">Page: ${pdfSection.page}</div>`;
    }
    if (pdfSection.content) {
        html += `<div class="pdf-content">${markdownToHtml(pdfSection.content)}</div>`;
    }
    if (pdfSection.table) {
        html += renderTable(pdfSection.table);
    }
    html += '</div>';
    return html;
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