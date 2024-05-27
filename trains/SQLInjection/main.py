import psycopg2
import os

print("Welcome to the PADME Playground Eval!")
print(os.environ["STATION_NAME"])

conn = psycopg2.connect(dbname="patientdb",
                        user=os.environ["DATA_SOURCE_USERNAME"],
                        host=os.environ["DATA_SOURCE_HOST"],
                        password=os.environ["DATA_SOURCE_PASSWORD"],
                        port=os.environ["DATA_SOURCE_PORT"])

cursor = conn.cursor()

cursor.execute(f"SELECT * FROM {'patient_info' if os.environ['STATION_NAME'] == 'Bruegel' else 'patients'} WHERE age >= 50")
result = cursor.fetchall()
nPatientsOver50 = len(result)

# emulate SQL injection
age = f"50; UPDATE {'patient_info' if os.environ['STATION_NAME'] == 'Bruegel' else 'patients'} SET age = 999 WHERE age >= 50"
cursor.execute(f"SELECT * FROM {'patient_info' if os.environ['STATION_NAME'] == 'Bruegel' else 'patients'} WHERE age >= {age}")


print(f"Number of rows affected: {nPatientsOver50}")

nPatientsOver50 = len(result)

# Write Dummy File
f = open("result.txt", "a")
f.write(f"{nPatientsOver50}\n")
f.close()