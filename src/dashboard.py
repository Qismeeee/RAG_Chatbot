import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3


def get_feedback_data():
    conn = sqlite3.connect('feedback.db')
    query = '''
        SELECT 
            CASE WHEN f.feedback_type = 1 THEN 'Like' ELSE 'Dislike' END as feedback_type,
            COUNT(*) as count,
            strftime('%Y-%m-%d', f.timestamp) as date
        FROM feedback f
        GROUP BY f.feedback_type, date
        ORDER BY date ASC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def calculate_accuracy():
    conn = sqlite3.connect('feedback.db')
    query = '''
        SELECT 
            COUNT(CASE WHEN feedback_type = 1 THEN 1 END) as likes,
            COUNT(*) as total
        FROM feedback
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty or df['total'].iloc[0] == 0:
        return 0

    return (df['likes'].iloc[0] / df['total'].iloc[0] * 100)


def get_top_questions():
    conn = sqlite3.connect('feedback.db')
    query = '''
        SELECT 
            ch.query,
            COUNT(f.id) as like_count
        FROM chat_history ch
        JOIN feedback f ON ch.id = f.chat_id
        WHERE f.feedback_type = 1
        GROUP BY ch.query
        ORDER BY like_count DESC
        LIMIT 5
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def generate_feedback_charts():
    # Accuracy Gauge
    accuracy = calculate_accuracy()
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        title={'text': "Accuracy"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#ff7f0e"},
                {'range': [50, 80], 'color': "#2ca02c"},
                {'range': [80, 100], 'color': "#1f77b4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': accuracy
            }
        }
    ))

    # Feedback Timeline
    df = get_feedback_data()
    if not df.empty:
        timeline = px.bar(df,
                          x='date',
                          y='count',
                          color='feedback_type',
                          title='Feedback Over Time',
                          labels={'count': 'Number of Feedbacks'},
                          color_discrete_map={'Like': 'green', 'Dislike': 'red'})
    else:
        timeline = go.Figure()
        timeline.add_annotation(text="No feedback data available",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False)

    # Top Questions
    top_questions = get_top_questions()
    if not top_questions.empty:
        top_chart = px.bar(top_questions,
                           x='query',
                           y='like_count',
                           title='Most Liked Questions',
                           labels={'query': 'Question', 'like_count': 'Likes'},
                           color_discrete_sequence=['green'])
    else:
        top_chart = go.Figure()
        top_chart.add_annotation(text="No questions data available",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)

    return gauge, timeline, top_chart
