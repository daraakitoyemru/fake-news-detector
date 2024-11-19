document.querySelector("#checkText").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        function: getSelectedText,
      },
      (results) => {
        if (results && results[0] && results[0].result) {
          // Handle the selected text here (e.g., send it to your backend API)

          const text = results[0].result;
          console.log("Selected text:", text); // Debug: Log the selected text

          // Make sure to call checkCredibility to process the text
          checkCredibility(text);
        } else {
          document.querySelector("result").textContent = "No text selected!";
        }
      }
    );
  });
});

// Function to run in the content script to get highlighted text
function getSelectedText() {
  return window.getSelection().toString();
}

// Function to make an API call to check text credibility
function checkCredibility(text) {
  // Replace with your GCP-deployed API URL
  fetch("https://fake-news-detector-5xvv.onrender.com/check", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: text }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Display the result in the popup
      document.getElementById("result").textContent = data.result;

      // If the text is fake, suggest credible sources
      if (data.probabilities.Fake > 0.9) {
        fetchCredibleSources(text); // Call the function to fetch credible sources
      }
    })
    .catch((error) => {
      document.getElementById("result").textContent = "Error: " + error.message;
    });
}

// Placeholder for NewsAPI integration
function fetchCredibleSources(text) {
  // Implementation to fetch credible sources using NewsAPI will go here
  document.getElementById("credibleSources").textContent =
    "Fetching credible sources...";
}
