// Select elements
const loadingDiv = document.getElementById("loading");
const resultDiv = document.getElementById("result");
const sourcesDiv = document.getElementById("credibleSources");

const newsAPIKey = CONFIG.NEWS_API_KEY;

// Function to show the loading spinner
function showLoading() {
  loadingDiv.style.display = "block";
  resultDiv.style.display = "none";
  sourcesDiv.style.display = "none";
}

// Function to hide the loading spinner
function hideLoading() {
  loadingDiv.style.display = "none";
  resultDiv.style.display = "block";
  sourcesDiv.style.display = "block";
}

// Event listener for the "Check Credibility" button
document.querySelector("#checkText").addEventListener("click", async () => {
  try {
    console.log("Check Credibility button clicked"); // Debugging
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const results = await chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      function: getSelectedText,
    });

    if (results && results[0] && results[0].result) {
      const text = results[0].result;
      console.log("Selected text:", text); // Debugging
      showLoading(); // Show loading spinner
      await Promise.all([checkCredibility(text), fetchCredibleSources(text)]);
    } else {
      document.querySelector("#result").textContent = "No text selected!";
    }
  } catch (error) {
    console.error("Error processing text:", error);
    document.querySelector("#result").textContent = "Error processing request.";
  } finally {
    hideLoading(); // Ensure spinner hides after processing
  }
});

// Function to get selected text from the webpage
function getSelectedText() {
  return window.getSelection().toString();
}

// Function to check the credibility of the selected text
async function checkCredibility(text) {
  try {
    const response = await fetch(
      "https://fake-news-detector-5xvv.onrender.com/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    console.log("API response:", data); // Debugging

    if (data.summary_verdict && data.summary_verdict.explanation) {
      resultDiv.textContent = data.summary_verdict.explanation;
    } else {
      resultDiv.textContent = "Unable to find relevant results.";
    }
  } catch (error) {
    console.error("API error:", error); // Debugging
    resultDiv.textContent = "Error: Could not fetch data.";
  }
}

// Function to fetch credible sources for the selected text
async function fetchCredibleSources(text) {
  try {
    const response = await fetch(
      `https://newsapi.org/v2/everything?q=${encodeURIComponent(
        text
      )}&language=en&sortBy=relevancy`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${newsAPIKey}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    const sources = data.articles || [];
    sourcesDiv.innerHTML = ""; // Clear previous sources

    if (sources.length > 0) {
      sources.forEach((article) => {
        const sourceItem = document.createElement("div");
        sourceItem.innerHTML = `<a href="${article.url}" target="_blank">${article.title}</a>`;
        sourcesDiv.appendChild(sourceItem);
      });
    } else {
      sourcesDiv.textContent =
        "No credible sources found. This has to do with the API and the format of search criteria.";
    }
  } catch (error) {
    console.error("Error fetching sources:", error);
    sourcesDiv.textContent = "Error fetching credible sources.";
  }
}
