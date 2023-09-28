import pymysql
import pandas as pd

conn = pymysql.connect(
    host='localhost',
    user='master',
    password='thesis23',
    db='masterthesis',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

df = pd.read_csv('/Users/quintengroenveld/Downloads/flickr_images.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
df = df.fillna(0)
try:
    with conn.cursor() as cursor:
        for i, r in df.iterrows():
            print(i)
            # Create a new record
            sql = "INSERT INTO `swissimages` (`id`, `latitude`, `longitude`, `post_create_date`, `post_publish_date`, `url`, `post_views`, `post_like_count`, `tags`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (str(r['post_guid']), float(r['latitude']), float(r['longitude']), r['post_create_date'], r['post_publish_date'], str(r['post_thumbnail_url']), int(r['post_views_count']), int(r['post_like_count']), str(r['tags'])))

    # Commit changes
    conn.commit()

    print("Record inserted successfully")
finally:
    conn.close()

