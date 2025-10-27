# Project_SIPRec-TW: SIP-Aware Social Sentiment Recommendation System for Foodpanda Taiwan

專案簡介，本專案結合 __自然語言處理（NLP）__ 技術與 __深度學習（LSTM 模型）__ 分析 Foodpanda 台灣平台上的用戶評論，提取語意與情緒特徵，建立能感知使用者情緒與社交語境的美食推薦系統。

## 「SIP-Aware 台灣美食推薦系統」核心特色

 - 情緒分類模型：建立 LSTM 模型進行情緒標記（正面、中性、負面）。
 - SIP 特徵擷取：分析評論中的第一人稱使用、情感詞、emoji、社交語句等。
 - 評論語義向量化：提取評論的語意向量作為推薦基礎。
 - 個人化推薦：根據Foodpanda 台灣平台使用者過往評論風格，推薦適合的餐廳。

 
## 「SIP-Aware 台灣美食推薦系統」推薦結果範例

推薦給使用者 y4fsj0dh 會有情感連結的前 20 家餐廳如下：  

| #  | StoreId | CompleteStoreName                                 | FoodType | City            | Location                 | Similarity |
| -- | ------- | ------------------------------------------------- | -------- | --------------- | ------------------------ | ---------- |
| 1  | zzvo    | 也鰻仁日式料理店                                          | 日式       | taipei-city     | NaN                      | 0.0        |
| 2  | a06a    | 三媽臭臭鍋 (台北內湖店)                                     | 台式       | taipei-city     | NaN                      | 0.0        |
| 3  | a07o    | 林森79熱炒 (喫酒夜攤)                                     | 台式       | new-taipei-city | NaN                      | 0.0        |
| 4  | a08d    | Champion Leg Rice                                 | Desktop  | tainan          | Champion Leg Rice        | 0.0        |
| 5  | a08d    | 狀元腿庫飯                                             | 台式       | tainan          | NaN                      | 0.0        |
| 6  | a098    | 燒鳥串道日式串燒                                          | 日式       | new-taipei-city | NaN                      | 0.0        |
| 7  | a09u    | 南洋冰品店                                             | 甜點       | taichung        | NaN                      | 0.0        |
| 8  | a0b4    | 綠蔬事務所                                             | 健康餐      | tainan          | 綠蔬事務所                    | 0.0        |
| 9  | a0b4    | 綠蔬事務所                                             | 健康餐      | tainan          | NaN                      | 0.0        |
| 10 | a0bl    | 隨主飡法式水煮專賣 (台南富農店)                                 | 健康餐      | tainan          | NaN                      | 0.0        |
| 11 | a0bp    | 光頭早午餐                                             | 早餐       | new-taipei-city | NaN                      | 0.0        |
| 12 | zxo2    | Suke Dog 吃樂狗以捏 (台北內湖店)                            | 歐美       | taipei-city     | NaN                      | 0.0        |
| 13 | zxqx    | blue磚塊廚房 (台北西門店)                                  | 歐美       | taipei-city     | 台北西門店                    | 0.0        |
| 14 | zxqx    | blue磚塊廚房 (台北西門店)                                  | 歐美       | taipei-city     | NaN                      | 0.0        |
| 15 | zxrx    | 正宗福隆便當 (新北土城店)                                    | 台式       | new-taipei-city | NaN                      | 0.0        |
| 16 | zxrx    | Authentic Fulong Bento (New Taipei Tucheng Store) | Desktop  | new-taipei-city | New Taipei Tucheng Store | 0.0        |
| 17 | zxv8    | 家中蔬食Vegetarian House                              | 素食       | taichung        | NaN                      | 0.0        |
| 18 | zxwa    | 貓子雞蛋糕                                             | 甜點       | taichung        | NaN                      | 0.0        |
| 19 | zxzq    | 兄弟鹽酥雞 (台北東湖店)                                     | 小吃       | taipei-city     | 台北東湖店                    | 0.0        |
| 20 | zxe4    | 雞嚐LULU鹽水雞                                         | 小吃       | tainan          | NaN                      | 0.0        |


