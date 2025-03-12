from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table
from sqlalchemy.orm import relationship
from config import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    project_id = Column(String)
    story_points = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Define relationships
    user = relationship('User', back_populates='tasks')
    skills = relationship('TaskSkill', back_populates='task')
    assignment = relationship('Assignment', back_populates='task', uselist=False)

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True)
    role = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Define relationships
    user = relationship('User', back_populates='employees')
    skills = relationship('EmployeeSkill', back_populates='employee')
    assignments = relationship('Assignment', back_populates='employee')

class TaskSkill(Base):
    __tablename__ = 'task_skills'
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey('tasks.id'))
    skill_name = Column(String)
    skill_level = Column(Float)
    
    # Define relationship
    task = relationship('Task', back_populates='skills')

class EmployeeSkill(Base):
    __tablename__ = 'employee_skills'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    skill_name = Column(String)
    skill_level = Column(Float)
    
    # Define relationship
    employee = relationship('Employee', back_populates='skills')

class Assignment(Base):
    __tablename__ = 'assignments'
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey('tasks.id'), unique=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    similarity_score = Column(Float)
    status = Column(String, default='To Do')
    
    # Define relationships
    task = relationship('Task', back_populates='assignment')
    employee = relationship('Employee', back_populates='assignments')

# Add back-references to User model
User.tasks = relationship('Task', back_populates='user')
User.employees = relationship('Employee', back_populates='user')