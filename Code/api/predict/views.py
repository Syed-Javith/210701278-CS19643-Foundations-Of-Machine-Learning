from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from twilio.rest import Client
from django.conf import settings
@api_view(['POST'])
def process_data(request):
    try:
        data = request.data
        
        df = pd.DataFrame(data) 

        df.drop(['phone_no', 'customer_id'], axis=1, inplace=True)
        df['gender'] = df['gender'].map(lambda x: 1 if x == 'Male' else 0)
        df['multi_screen'] = df['multi_screen'].apply(lambda x: 1 if x == 'yes' else 0)
        df['mail_subscribed'] = df['mail_subscribed'].apply(lambda x: 1 if x == 'yes' else 0)
        
        X = df.drop(['year'], axis=1)
        
        scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.joblib'))
        X_scaled = scaler.transform(X)

        model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        model = joblib.load(model_path)
        
        pred = model.predict(X_scaled)
        print(pred[0])
        if(pred[0]==1):
            client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        # Send the SMS
            message = client.messages.create(
                body="Hello, We are from XYZ Company, we will be happy if you continue your subscription.",
                from_=settings.TWILIO_PHONE_NUMBER,
                to="+916380411427"
            )
            print(message.sid)
        return Response({
            'X_scaled': X_scaled.tolist(), 
            'prediction': pred.tolist()
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)
