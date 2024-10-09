const express = require('express');
const cors = require('cors');
const { PDFLoader } = require("@langchain/community/document_loaders/fs/pdf");
const { ChatOpenAI } = require("@langchain/openai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { createStuffDocumentsChain } = require("langchain/chains/combine_documents");
const { createRetrievalChain } = require("langchain/chains/retrieval");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
require('dotenv').config();
const { ChatGoogleGenerativeAI } =require("@langchain/google-genai");

const model = new ChatGoogleGenerativeAI({
  model: "gemini-pro",
  maxOutputTokens: 2048,
});


const app = express();
const port = 3000;

// Enable CORS for all origins
app.use(cors());

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.error("Please set your OPENAI_API_KEY in your environment.");
  process.exit(1);
}

let retriever; // This will hold the retriever created at server startup

// Load and prepare the PDF on server startup
async function initializeDocumentProcessing(pdfPath) {
  try {
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splits = await textSplitter.splitDocuments(docs);

    const vectorstore = await MemoryVectorStore.fromDocuments(
      splits,
      new OpenAIEmbeddings({ apiKey: OPENAI_API_KEY })
    );

    // Set up the retriever for later use
    retriever = vectorstore.asRetriever();
    console.log("PDF loaded and vectorized successfully.");
  } catch (error) {
    console.error("Error loading and processing PDF:", error);
  }
}

// Function to handle question answering based on the preloaded document
async function answerQuestion(question) {
  try {
    if (!retriever) {
      throw new Error("Document retriever is not ready.");
    }

    const systemTemplate = [
      `You are an assistant for question-answering tasks. you only answer in marathi langaguage no matter what  `,
      `Use the following pieces of retrieved context to answer `,
      `the question. If you don't know the answer, say that you `,
      `don't know. Use three sentences maximum and keep the `,
      `answer concise. `,
      `\n\n`,
      `{context}`,
    ].join("");

    const prompt = ChatPromptTemplate.fromMessages([
      ["human", systemTemplate],
      ["human", "{input}"],
    ]);

    // const model = new ChatOpenAI({ apiKey: OPENAI_API_KEY, model: "gpt-4" });

    const questionAnswerChain = await createStuffDocumentsChain({
      llm: model,
      prompt,
    });

    const ragChain = await createRetrievalChain({
      retriever,
      combineDocsChain: questionAnswerChain,
    });

    const results = await ragChain.invoke({
      input: question,
    });

    return results.answer;
  } catch (error) {
    console.error("Error during question answering:", error);
    return "An error occurred while processing your request.";
  }
}

// Define the GET route for answering questions
app.get('/ask', async (req, res) => {
  let question = req.query.question;

  if (!question) {
    return res.status(400).send("Please provide a question.");
  }
  question =await model.invoke(["human","answer the following question in marathi only answer shoudl be in one paragraph "+ question]);
  const answer=question['content'];
  res.send({ answer });
});

// Start the server and initialize document processing
app.listen(port, async () => {
  console.log(`Server running on port ${port}`);
  // await initializeDocumentProcessing('Handbook_ICAR_16.04.2020.pdf');
});
