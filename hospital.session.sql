CREATE TABLE Patients (
    PatientID SERIAL PRIMARY KEY,  -- Auto-incrementing Primary Key
    FullName VARCHAR(100) NOT NULL,            -- Patient's full name
    DOB DATE NOT NULL,                         -- Date of birth (cannot be null)
    Gender VARCHAR(10) NOT NULL,               -- Gender (e.g., Male, Female)
    ContactNo VARCHAR(15) NOT NULL,            -- Contact number (cannot be null)
    Address VARCHAR(255),                      -- Patient's address
    BloodGroup VARCHAR(10),                    -- Patient's blood group
    MedicalHistory TEXT                        -- Medical history of the patient
);

CREATE TABLE Departments (
    DepartmentID SERIAL PRIMARY KEY,  -- Auto-incrementing Primary Key
    DepartmentName VARCHAR(100) NOT NULL           -- Department name (e.g., Cardiology)
);


CREATE TABLE Doctors (
    DoctorID SERIAL PRIMARY KEY,   -- Auto-incrementing Primary Key
    FullName VARCHAR(100) NOT NULL,             -- Doctor's full name
    Specialization VARCHAR(100) NOT NULL,       -- Doctor's specialization (e.g., Cardiologist)
    PhoneNo VARCHAR(15) NOT NULL,               -- Doctor's phone number
    Email VARCHAR(100),                         -- Doctor's email
    DepartmentID INT,                           -- Foreign key to Departments table
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID) ON DELETE SET NULL
);

CREATE TABLE Appointments (
    AppointmentID SERIAL PRIMARY KEY,  -- Auto-incrementing Primary Key
    PatientID INT NOT NULL,                        -- Foreign key to Patients table
    DoctorID INT NOT NULL,                         -- Foreign key to Doctors table
    AppointmentDate DATE NOT NULL,                 -- Appointment date
    TimeSlot VARCHAR(50) NOT NULL,                 -- Appointment time slot
    Status VARCHAR(20) DEFAULT 'Scheduled',        -- Appointment status (Scheduled, Completed, etc.)
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE, -- Cascade delete
    FOREIGN KEY (DoctorID) REFERENCES Doctors(DoctorID) ON DELETE SET NULL  -- Set to NULL on doctor deletion
);

CREATE TABLE MedicalRecords (
    RecordID SERIAL PRIMARY KEY,      -- Auto-incrementing Primary Key
    PatientID INT NOT NULL,                        -- Foreign key to Patients table
    DoctorID INT NOT NULL,                         -- Foreign key to Doctors table
    Diagnosis TEXT NOT NULL,                       -- Diagnosis information
    Prescription TEXT,                             -- Prescription details
    Date DATE NOT NULL,                            -- Date when the record was created
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE,  -- Cascade delete
    FOREIGN KEY (DoctorID) REFERENCES Doctors(DoctorID) ON DELETE SET NULL   -- Set to NULL on doctor deletion
);

CREATE TABLE Bills (
    BillID SERIAL PRIMARY KEY,        -- Auto-incrementing Primary Key
    PatientID INT NOT NULL,                        -- Foreign key to Patients table
    Amount DECIMAL(10, 2) NOT NULL,                -- Total bill amount
    BillDate DATE NOT NULL,                        -- Date of the bill
    PaymentStatus VARCHAR(20) DEFAULT 'Pending',   -- Payment status (Pending, Paid)
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE  -- Cascade delete
);

CREATE TABLE Tests (
    TestID SERIAL PRIMARY KEY,         -- Auto-incrementing Primary Key
    PatientID INT NOT NULL,                         -- Foreign key to Patients table
    DoctorID INT NOT NULL,                          -- Foreign key to Doctors table
    TestType VARCHAR(100) NOT NULL,                 -- Type of test (e.g., Blood Test, X-ray)
    TestResult TEXT NOT NULL,                       -- Result of the test
    TestDate DATE NOT NULL,                         -- Date when the test was performed
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE,   -- Cascade delete
    FOREIGN KEY (DoctorID) REFERENCES Doctors(DoctorID) ON DELETE SET NULL     -- Set to NULL on doctor deletion
);


INSERT INTO Departments (DepartmentName) VALUES 
('Cardiology'),
('Neurology'),
('Orthopedics'),
('General Medicine'),
('Pediatrics');

INSERT INTO Doctors (FullName, Specialization, PhoneNo, Email, DepartmentID) VALUES 
('Dr. Ahmed Khan', 'Cardiologist', '03001234567', 'ahmed.khan@hospital.pk', 1),
('Dr. Sarah Ali', 'Neurologist', '03111234567', 'sarah.ali@hospital.pk', 2),
('Dr. Bilal Hussain', 'Orthopedic Surgeon', '03221234567', 'bilal.hussain@hospital.pk', 3),
('Dr. Ayesha Malik', 'General Physician', '03331234567', 'ayesha.malik@hospital.pk', 4),
('Dr. Muhammad Usman', 'Pediatrician', '03451234567', 'usman.pediatrics@hospital.pk', 5);

INSERT INTO Patients (FullName, DOB, Gender, ContactNo, Address, BloodGroup, MedicalHistory) VALUES 
('Ali Raza', '1990-08-12', 'Male', '03034567891', 'House #12, Street #5, Lahore', 'B+', 'Hypertension'),
('Fatima Noor', '1985-03-25', 'Female', '03125678902', 'Plot #23, Clifton Block 4, Karachi', 'O-', 'Diabetes'),
('Hamza Sheikh', '2000-06-10', 'Male', '03216789034', 'House #77, G-9, Islamabad', 'A+', 'Asthma'),
('Aisha Yousaf', '1998-11-20', 'Female', '03337890145', 'Apartment #3, Gulberg, Lahore', 'AB+', 'No major history'),
('Raza Hussain', '1975-01-15', 'Male', '03458901267', 'Sector C, Bahria Town, Rawalpindi', 'B-', 'Heart Disease');

INSERT INTO Appointments (PatientID, DoctorID, AppointmentDate, TimeSlot, Status) VALUES 
(1, 1, '2024-04-10', '10:00 AM', 'Scheduled'),
(2, 2, '2024-04-12', '11:30 AM', 'Scheduled'),
(3, 3, '2024-04-15', '09:45 AM', 'Completed'),
(4, 4, '2024-04-20', '02:00 PM', 'Cancelled'),
(5, 5, '2024-04-25', '03:30 PM', 'Scheduled');

INSERT INTO MedicalRecords (PatientID, DoctorID, Diagnosis, Prescription, Date) VALUES 
(1, 1, 'High Blood Pressure', 'Losartan 50mg', '2024-04-10'),
(2, 2, 'Migraine', 'Sumatriptan 100mg', '2024-04-12'),
(3, 3, 'Fractured Arm', 'Ibuprofen 400mg', '2024-04-15'),
(4, 4, 'Fever & Weakness', 'Paracetamol 500mg', '2024-04-20'),
(5, 5, 'Childhood Asthma', 'Salbutamol Inhaler', '2024-04-25');

INSERT INTO Bills (PatientID, Amount, BillDate, PaymentStatus) VALUES 
(1, 5000, '2024-04-10', 'Paid'),
(2, 7000, '2024-04-12', 'Pending'),
(3, 12000, '2024-04-15', 'Paid'),
(4, 4000, '2024-04-20', 'Cancelled'),
(5, 8000, '2024-04-25', 'Pending');

INSERT INTO Tests (PatientID, DoctorID, TestType, TestResult, TestDate) VALUES 
(1, 1, 'Blood Pressure Test', 'High BP detected', '2024-04-10'),
(2, 2, 'MRI Scan', 'No abnormality detected', '2024-04-12'),
(3, 3, 'X-ray', 'Fracture in left arm', '2024-04-15'),
(4, 4, 'Blood Test', 'Normal', '2024-04-20'),
(5, 5, 'Lung Function Test', 'Asthma detected', '2024-04-25');

SELECT * FROM appointments;
TRUNCATE TABLE Patients RESTART IDENTITY CASCADE;


ALTER TABLE Doctors
ADD COLUMN DoctorFee DECIMAL(10,2) NOT NULL DEFAULT 0.00,
ADD COLUMN DoctorSalary DECIMAL(10,2) NOT NULL DEFAULT 0.00;

UPDATE Doctors
SET DoctorFee = 1500.00, DoctorSalary = 150000.00
WHERE FullName = 'Dr. Ahmed Khan';

UPDATE Doctors
SET DoctorFee = 2000.00, DoctorSalary = 180000.00
WHERE FullName = 'Dr. Sarah Ali';

UPDATE Doctors
SET DoctorFee = 1800.00, DoctorSalary = 170000.00
WHERE FullName = 'Dr. Bilal Hussain';

UPDATE Doctors
SET DoctorFee = 1000.00, DoctorSalary = 120000.00
WHERE FullName = 'Dr. Ayesha Malik';

UPDATE Doctors
SET DoctorFee = 1300.00, DoctorSalary = 140000.00
WHERE FullName = 'Dr. Muhammad Usman';

ALTER TABLE MedicalRecords 
ADD COLUMN MedicineFee DECIMAL(10,2) DEFAULT 0.00;

UPDATE MedicalRecords 
SET MedicineFee = 500.00 -- Example fee
WHERE RecordID = 1;

UPDATE MedicalRecords 
SET MedicineFee = 300.00 -- Example fee
WHERE RecordID = 2;

UPDATE MedicalRecords 
SET MedicineFee = 700.00 -- Example fee
WHERE RecordID = 3;

UPDATE MedicalRecords 
SET MedicineFee = 250.00 -- Example fee
WHERE RecordID = 4;

UPDATE MedicalRecords 
SET MedicineFee = 450.00 -- Example fee
WHERE RecordID = 5;

ALTER TABLE Tests 
ADD COLUMN TestPrice DECIMAL(10,2) DEFAULT 0.00;

UPDATE Tests 
SET TestPrice = 500.00  -- Example price for Blood Pressure Test
WHERE TestType = 'Blood Pressure Test';

UPDATE Tests 
SET TestPrice = 8000.00  -- Example price for MRI Scan
WHERE TestType = 'MRI Scan';

UPDATE Tests 
SET TestPrice = 3000.00  -- Example price for X-ray
WHERE TestType = 'X-ray';

UPDATE Tests 
SET TestPrice = 1500.00  -- Example price for Blood Test
WHERE TestType = 'Blood Test';

UPDATE Tests 
SET TestPrice = 4000.00  -- Example price for Lung Function Test
WHERE TestType = 'Lung Function Test';

ALTER TABLE Bills 
ADD COLUMN Expenses DECIMAL(10, 2) DEFAULT 0.00;

SELECT * FROM bills
ALTER TABLE Bills DROP COLUMN Amount;


SELECT * FROM doctors

Drop the existing foreign key constraint on DoctorID in Appointments table
ALTER TABLE Appointments
DROP CONSTRAINT IF EXISTS DoctorID;

ALTER TABLE Appointments
ADD CONSTRAINT DoctorID
FOREIGN KEY (DoctorID) REFERENCES Doctors(DoctorID) ON DELETE SET NULL;


ALTER TABLE appointments
ALTER COLUMN doctorid DROP NOT NULL;

DELETE FROM doctors WHERE doctorid = 3;

ALTER TABLE medicalrecords
ALTER COLUMN doctorid DROP NOT NULL;

ALTER TABLE medicalrecords
DROP CONSTRAINT IF EXISTS medicalrecords_doctorid_fkey,
ADD CONSTRAINT medicalrecords_doctorid_fkey
FOREIGN KEY (doctorid) REFERENCES doctors(doctorid) ON DELETE SET NULL;

ALTER TABLE tests
ALTER COLUMN doctorid DROP NOT NULL;

ALTER TABLE tests
DROP CONSTRAINT IF EXISTS tests_doctorid_fkey,
ADD CONSTRAINT tests_doctorid_fkey
FOREIGN KEY (doctorid) REFERENCES doctors(doctorid) ON DELETE SET NULL;


ALTER TABLE MedicalRecords 
ADD COLUMN MedicineFee DECIMAL(10,2) DEFAULT 0.00;

CREATE TABLE pharmacy (
    pharmacy_id SERIAL PRIMARY KEY,  -- Unique ID for each medicine entry
    medicine_name VARCHAR(255) ,  -- Name of the medicine
    quantity INT,  -- Quantity of the medicine in stock
    price_per_unit DECIMAL(10, 2) NOT NULL,  -- Price per unit of the medicine (with 2 decimal places)
    PatientID INT NOT NULL,  -- Foreign key or identifier for the patient related to this medicine entry
    FOREIGN KEY (PatientID) REFERENCES patients(PatientID)  -- Assuming there is a 'patients' table with 'id' as the primary key
);

Insert sample medicines into the pharmacy table
INSERT INTO pharmacy (medicine_name, quantity, price_per_unit, PatientID) VALUES
('Aspirin', 50, 10.50, 2),  -- John Doe (patient_id = 1)
('Paracetamol', 100, 5.75, 5),  -- Jane Smith (patient_id = 2)
('Amoxicillin', 30, 12.30, 13);  -- Emily Johnson (patient_id = 3)

Rename columns in the pharmacy table to remove underscores
ALTER TABLE pharmacy RENAME COLUMN medicine_name TO medicinename;
ALTER TABLE pharmacy RENAME COLUMN price_per_unit TO priceperunit;

ALTER TABLE Tests
DROP COLUMN DoctorID;


Step 1: Drop the existing Bills table (if it exists)
DROP TABLE IF EXISTS Bills;

Step 2: Create the Bills table with the new columns
CREATE TABLE Bills (
    BillID SERIAL PRIMARY KEY,        -- Auto-incrementing Primary Key
    PatientID INT NOT NULL,                        -- Foreign key to Patients table
    BillDate DATE NOT NULL,                        -- Date of the bill
    PaymentStatus VARCHAR(20) DEFAULT 'Pending',   -- Payment status (Pending, Paid)
    Expenses DECIMAL(10, 2) NOT NULL,              -- Total expenses (bill amount)
    FOREIGN KEY (PatientID) REFERENCES Patients(PatientID) ON DELETE CASCADE  -- Cascade delete
);

INSERT INTO doctors (FullName, Specialization, PhoneNo, Email, DepartmentID, DoctorFee, DoctorSalary)
VALUES 
('Dr. Ali Khan', 'Cardiology', '+92-300-1234567', 'dr.ali.khan@email.com', 1, 1500, 50000),
('Dr. Sara Iqbal', 'Neurology', '+92-301-2345678', 'dr.sara.iqbal@email.com', 2, 1800, 55000),
('Dr. Ahmed Shah', 'Orthopedics', '+92-302-3456789', 'dr.ahmed.shah@email.com', 3, 1200, 60000),
('Dr. Fatima Baig', 'Pediatrics', '+92-303-4567890', 'dr.fatima.baig@email.com', 4, 1300, 52000),
('Dr. Imran Abbas', 'Dermatology', '+92-304-5678901', 'dr.imran.abbas@email.com', 2, 1400, 53000),
('Dr. Mehwish Noor', 'Gynecology', '+92-305-6789012', 'dr.mehwish.noor@email.com', 3, 1600, 54000),
('Dr. Zain Ahmed', 'General Surgery', '+92-306-7890123', 'dr.zain.ahmed@email.com', 4, 1700, 58000);

INSERT INTO Appointments (PatientID, DoctorID, AppointmentDate, TimeSlot, Status)
VALUES
(5, 16, '2025-05-10', '10:00 AM - 11:00 AM', 'Scheduled'),
(2, 17, '2025-05-10', '11:00 AM - 12:00 PM', 'Scheduled'),
(13, 18, '2025-05-11', '12:00 PM - 01:00 PM', 'Scheduled'),
(16, 19, '2025-05-12', '02:00 PM - 03:00 PM', 'Scheduled'),
(17, 20, '2025-05-13', '03:00 PM - 04:00 PM', 'Scheduled');

INSERT INTO MedicalRecords (PatientID, DoctorID, Diagnosis, Prescription, Date) VALUES
(5, 16, 'High Blood Pressure', 'Losartan 50mg', '2024-04-10'),
(2, 17, 'Migraine', 'Sumatriptan 100mg', '2024-04-12'),
(13, 18, 'Fractured Arm', 'Ibuprofen 400mg', '2024-04-15'),
(16, 19, 'Fever & Weakness', 'Paracetamol 500mg', '2024-04-20'),
(17, 20, 'Childhood Asthma', 'Salbutamol Inhaler', '2024-04-25');
