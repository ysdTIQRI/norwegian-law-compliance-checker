
import psycopg

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "norwegian_law_checker",
    "user": "postgres",
    "password": "postgres"
}

try:
    with psycopg.connect(**DB_CONFIG) as connection:
        with connection.cursor() as cursor:

            # Test PostgreSQL connection
            cursor.execute("SELECT version();")
            result = cursor.fetchone()

            print("PostgreSQL connection successful!")
            print(result[0])

            # Test pgvector extension
            cursor.execute("""
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname = 'vector';
            """)

            vector = cursor.fetchone()

            if vector:
                print("\npgvector connection successful!")
                print("Extension:", vector[0])
                print("Version:", vector[1])
            else:
                print("\npgvector extension not found!")

except Exception as error:
    print("PostgreSQL connection failed:")
    print(error)
