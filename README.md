# Project_SIPRec-TW: SIP-Aware Social Sentiment Recommendation System for Foodpanda Taiwan

專案簡介，本專案結合 __自然語言處理（NLP）__ 技術與 __社會資訊處理理論（Social Information Processing Theory, SIP）__，透過深度學習（LSTM 模型）分析 Foodpanda 台灣平台上的用戶評論，提取語意與情緒特徵，建立能感知使用者情緒與社交語境的美食推薦系統。

🔍 核心特色

 - 情緒分類模型：使用 LSTM 模型進行情緒標記（正面、中性、負面）；LSTM 是一種神經網路模型，而且是遞迴式神經網路（RNN, Recurrent Neural Network）的一種改良版本，特別適合處理具有時間序列特性的資料，如文字、語言與評論等內容。
 - SIP 特徵擷取：分析評論中的第一人稱使用、情感詞、emoji、社交語句等
 - 評論語義向量化：提取評論的語意向量作為推薦基礎
 - 個人化推薦：根據Foodpanda 台灣平台使用者過往評論風格，推薦適合的餐廳
