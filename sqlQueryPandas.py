import pymysql
conn = pymysql.connect(host='127.0.0.1',
                       port=3306,
                       user='root', 
                       passwd='',  
                       db='bigram_schema',
                       charset='utf8')

cursor = conn.cursor()

sql = "INSERT INTO bigram_schema.`stemmed_29.07.2020.last_29k` (original_text, label, dictionaries_words_stem, 28k_words_stem, golang_algo, 76k_words_stem, english) VALUES (%s, %s, %s, %s, %s, %s, %s);"

for i in df.values:
    cursor.execute(sql, (i[0], i[1], i[2], i[3], i[4], i[5], i[6]))

conn.commit()
conn.close()