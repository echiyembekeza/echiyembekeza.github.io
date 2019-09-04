---
layout: post
title: The Crux of the Opioid Crisis
description: Healthcare Data Science Project 
---

# It’s clear that there is a huge problem with opioid use in the United States.

While that is evident, the debate comes when we are discussing the reason for the crisis.Having a healthcare administrator perspective, I am always curious about the "root cause" of issues. I know...I know...data scientists should never speak in absolutes like that. But I believe that when given a great deal of information, we should be able to make reasonable assumptions or projections.

I am a firm believer that "Data is good. Good data is better. But actionable data is best." We have data on millions of patient encounters from the emergency departments in the state of Florida. Each of these observations provides information on the patient's race, ethinicity, zip code, primary reason for presenting to the ED, time of arrival, and many other pertinent pieces of information.

The data scientist in me was thrilled to think about the possibility of being able to use this information to begin laying a foundation to shape the framework of potential interventions to the opioid crisis. Perhaps if we knew what the most important features were in determining whether a person would present to the ED for opioid-related issues, we could begin to combat them. That is exactly what I decided to do.

Patients who presented to the ED for opioid use are identified using ICD-10 codes. Drug-poisoning deaths are identified using underlying cause-of-death codes:
F11.01 - F11.99
T40.1X1S - T40.693S

