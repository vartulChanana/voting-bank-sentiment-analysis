import nltk
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Expanded emotion keywords with more terms and related words
emotion_keywords = {
    'happy': ['happy', 'joy', 'delight', 'pleased', 'smile', 'cheerful', 'excited', 'glad', 'thrilled', 'elated', 
              'ecstatic', 'content', 'satisfied', 'delighted', 'jubilant', 'blissful', 'enjoy', 'loving', 'wonderful', 
              'fantastic', 'great', 'awesome', 'excellent', 'amazing', 'good', 'positive', 'win', 'success', 'celebrate',
              'congratulations', 'proud', 'pleasure', 'laugh', 'grin', 'optimistic', 'hopeful', 'enthusiastic', 'like',
              'love', 'adore', 'appreciate', 'grateful', 'thankful', 'blessed', 'fortunate', 'lucky', 'perfect', 'best',
              'better', 'improved', 'progress', 'achievement', 'accomplished', 'relief', 'relieved', 'relaxed', 'calm',
              'peaceful', 'serene', 'tranquil', 'comfortable', 'cozy', 'warm', 'bright', 'sunny', 'radiant', 'glowing'],
              
    'sad': ['sad', 'cry', 'unhappy', 'sorrow', 'mourn', 'depressed', 'tear', 'grief', 'miserable', 'heartbroken', 
            'gloomy', 'melancholy', 'despair', 'disappointed', 'upset', 'regret', 'hopeless', 'devastated', 'tragic', 
            'hurt', 'pain', 'suffering', 'lonely', 'alone', 'abandoned', 'rejected', 'loss', 'lost', 'fail', 'failure',
            'unfortunate', 'sorry', 'apology', 'miss', 'missing', 'broken', 'down', 'blue', 'troubled', 'distressed',
            'anguish', 'agony', 'woe', 'pity', 'sympathy', 'empathy', 'compassion', 'condolence', 'console', 'comfort',
            'weep', 'sob', 'wail', 'lament', 'mourn', 'grieve', 'bereaved', 'bereft', 'forlorn', 'desolate', 'dismal',
            'bleak', 'somber', 'solemn', 'grave', 'serious', 'heavy', 'burden', 'weight', 'pressure', 'strain', 'stress'],
            
    'angry': ['angry', 'mad', 'irate', 'furious', 'annoyed', 'outraged', 'resentful', 'rage', 'hate', 'hatred', 
              'hostile', 'irritated', 'enraged', 'infuriated', 'frustrated', 'aggravated', 'bitter', 'indignant', 
              'livid', 'offended', 'provoked', 'resentment', 'disgusted', 'upset', 'agitated', 'temper', 'yell', 
              'shout', 'scream', 'fight', 'conflict', 'argument', 'dispute', 'complain', 'blame', 'criticize', 'attack',
              'fury', 'wrath', 'vexed', 'cross', 'displeased', 'dissatisfied', 'exasperated', 'incensed', 'inflamed',
              'antagonistic', 'belligerent', 'combative', 'confrontational', 'contentious', 'contrary', 'defiant',
              'rebellious', 'resistant', 'obstinate', 'stubborn', 'uncooperative', 'difficult', 'problematic', 'trouble'],
              
    'fear': ['fear', 'scared', 'afraid', 'terrified', 'panic', 'anxious', 'worried', 'nervous', 'dread', 'horror', 
             'terror', 'fright', 'alarmed', 'threatened', 'intimidated', 'frightened', 'petrified', 'apprehensive', 
             'uneasy', 'tense', 'stress', 'stressed', 'concern', 'concerned', 'paranoid', 'suspicious', 'doubt', 
             'uncertain', 'insecure', 'vulnerable', 'helpless', 'danger', 'dangerous', 'risk', 'threat', 'warning',
             'phobia', 'phobic', 'aversion', 'avoid', 'escape', 'flee', 'run', 'hide', 'cower', 'shrink', 'retreat',
             'withdraw', 'back away', 'hesitant', 'reluctant', 'timid', 'shy', 'bashful', 'embarrassed', 'mortified',
             'humiliated', 'ashamed', 'guilty', 'remorseful', 'regretful', 'sorry', 'apologetic', 'defensive'],
             
    'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled', 'unexpected', 'sudden', 
                 'wonder', 'awe', 'bewildered', 'dumbfounded', 'speechless', 'astounded', 'flabbergasted', 'disbelief', 
                 'unbelievable', 'incredible', 'extraordinary', 'remarkable', 'wow', 'omg', 'gosh', 'whoa', 'unexpected',
                 'unpredictable', 'revelation', 'reveal', 'discovery', 'discover', 'unimaginable', 'unexpected', 'unforeseen',
                 'unanticipated', 'unpredicted', 'surprising', 'shocking', 'striking', 'staggering', 'overwhelming',
                 'breathtaking', 'impressive', 'magnificent', 'marvelous', 'wonderful', 'wondrous', 'miraculous',
                 'phenomenal', 'sensational', 'spectacular', 'stupendous', 'tremendous', 'unreal', 'wild', 'crazy'],
                 
    'disgust': ['disgust', 'disgusted', 'repulsed', 'revolted', 'disgusting', 'gross', 'nasty', 'sick', 'nauseous', 
               'repulsive', 'repugnant', 'offensive', 'foul', 'vile', 'distasteful', 'unpleasant', 'horrible', 'awful', 
               'terrible', 'appalling', 'hideous', 'objectionable', 'loathsome', 'detestable', 'abhorrent', 'hateful',
               'despicable', 'contempt', 'disdain', 'scorn', 'reject', 'refuse', 'unwanted', 'aversion', 'dislike',
               'disapprove', 'disapproval', 'condemn', 'condemnation', 'denounce', 'denunciation', 'criticize',
               'criticism', 'judge', 'judgment', 'censure', 'blame', 'fault', 'accuse', 'accusation', 'charge']
}

# Add phrases that indicate emotions
emotion_phrases = {
    'happy': ['feel good', 'feeling good', 'made my day', 'best day', 'so happy', 'very happy', 'really happy', 
              'so glad', 'very glad', 'really glad', 'feel blessed', 'feeling blessed', 'feel fortunate', 'feeling fortunate',
              'looking forward', 'can\'t wait', 'excited about', 'excited for', 'love it', 'love this', 'love that',
              'great news', 'good news', 'positive outcome', 'positive result', 'positive development', 'well done',
              'good job', 'great job', 'excellent work', 'proud of', 'pleased with', 'delighted with', 'happy with',
              'satisfied with', 'content with', 'enjoy the', 'enjoying the', 'enjoyed the', 'appreciate the', 
              'grateful for', 'thankful for', 'thank you', 'thanks for', 'congratulations on', 'congrats on', 
              'well deserved', 'keep up', 'looking good', 'sounds good', 'feels right', 'perfect for', 'ideal for',
              'best choice', 'right choice', 'good choice', 'wise decision', 'smart move', 'brilliant idea'],
              
    'sad': ['feel sad', 'feeling sad', 'feel down', 'feeling down', 'feel blue', 'feeling blue', 'broke my heart', 
            'breaks my heart', 'breaking my heart', 'heart aches', 'heart is heavy', 'heavy heart', 'miss you', 
            'missing you', 'feel lonely', 'feeling lonely', 'feel alone', 'feeling alone', 'lost hope', 'losing hope',
            'gave up', 'giving up', 'bad news', 'terrible news', 'unfortunate news', 'tragic news', 'sad news',
            'deeply saddened', 'deeply regret', 'deeply sorry', 'heartfelt condolences', 'sincere condolences',
            'thoughts and prayers', 'difficult time', 'challenging time', 'tough time', 'hard time', 'struggling with',
            'suffered a', 'suffering from', 'in pain', 'with pain', 'painful to', 'hurts to', 'hurt by', 'wounded by',
            'damaged by', 'harmed by', 'let down', 'disappointed by', 'disappointed with', 'disappointed in',
            'regret the', 'regrettable', 'unfortunate', 'sadly', 'unfortunately', 'wish that', 'if only'],
            
    'angry': ['piss off', 'pissed off', 'fed up', 'had enough', 'so angry', 'very angry', 'really angry', 'makes me angry',
              'made me angry', 'making me angry', 'get out', 'shut up', 'how dare', 'not fair', 'unfair', 'not right',
              'this is ridiculous', 'this is absurd', 'this is outrageous', 'this is unacceptable', 'cannot accept',
              'will not tolerate', 'won\'t stand for', 'had it with', 'tired of', 'sick of', 'sick and tired',
              'enough is enough', 'last straw', 'crossed the line', 'gone too far', 'out of line', 'who do you think',
              'what the hell', 'what the heck', 'damn it', 'damn you', 'screw you', 'screw this', 'hate when',
              'hate that', 'hate this', 'hate it', 'hate them', 'hate him', 'hate her', 'can\'t believe they',
              'can\'t believe he', 'can\'t believe she', 'can\'t believe you', 'should be ashamed', 'shame on'],
              
    'fear': ['scared of', 'afraid of', 'terrified of', 'fear of', 'worried about', 'anxious about', 'concerned about',
             'don\'t know what to do', 'what if', 'can\'t handle', 'can\'t cope', 'too much', 'help me',
             'frightens me', 'frightening', 'terrifies me', 'terrifying', 'scares me', 'scary', 'alarming', 'disturbing',
             'troubling', 'unsettling', 'upsetting', 'distressing', 'dreadful', 'horrible', 'awful', 'terrible',
             'worried that', 'worry that', 'concerned that', 'fear that', 'afraid that', 'scared that', 'terrified that',
             'anxious that', 'nervous about', 'nervous that', 'on edge', 'tense about', 'stressed about', 'stressed over',
             'freaking out', 'freaked out', 'losing sleep', 'lost sleep', 'can\'t sleep', 'nightmare', 'panic attack',
             'anxiety attack', 'heart racing', 'heart pounding', 'sweating', 'trembling', 'shaking', 'frozen'],
             
    'surprised': ['can\'t believe', 'cannot believe', 'hard to believe', 'never expected', 'didn\'t expect', 'did not expect',
                 'who knew', 'no way', 'never thought', 'never imagined', 'blew my mind', 'mind blown',
                 'took me by surprise', 'came as a surprise', 'surprising development', 'surprising turn', 'surprising outcome',
                 'surprising result', 'surprising discovery', 'surprising revelation', 'surprising finding', 'surprising fact',
                 'surprisingly', 'to my surprise', 'to my amazement', 'to my astonishment', 'to my shock', 'shocked to',
                 'amazed to', 'astonished to', 'stunned to', 'startled to', 'surprised to', 'unexpected turn',
                 'unexpected development', 'unexpected outcome', 'unexpected result', 'unexpected discovery',
                 'unexpected revelation', 'unexpected finding', 'unexpected fact', 'unexpectedly', 'out of nowhere',
                 'out of the blue', 'all of a sudden', 'suddenly realized', 'suddenly understood', 'just realized'],
                 
    'disgust': ['make me sick', 'makes me sick', 'made me sick', 'making me sick', 'can\'t stand', 'cannot stand',
               'hate it when', 'hate when', 'so gross', 'very gross', 'really gross', 'so disgusting', 'very disgusting',
               'really disgusting', 'absolutely disgusting', 'completely disgusting', 'totally disgusting', 'utterly disgusting',
               'thoroughly disgusting', 'grossed out', 'grosses me out', 'grossing me out', 'turns my stomach',
               'turned my stomach', 'turning my stomach', 'makes me nauseous', 'made me nauseous', 'making me nauseous',
               'makes me vomit', 'made me vomit', 'making me vomit', 'makes me gag', 'made me gag', 'making me gag',
               'repulsive behavior', 'repulsive action', 'repulsive attitude', 'repulsive comment', 'repulsive remark',
               'repulsive statement', 'repulsive sight', 'repulsive smell', 'repulsive taste', 'repulsive feeling',
               'revolting behavior', 'revolting action', 'revolting attitude', 'revolting comment', 'revolting remark']
}

# Add political/election specific emotion keywords
political_emotion_keywords = {
    'happy': ['support', 'endorse', 'back', 'advocate', 'promote', 'champion', 'defend', 'uphold', 'approve', 'favor', 
              'agree', 'concur', 'vote for', 'elect', 'victory', 'win', 'triumph', 'succeed', 'accomplish', 'achieve',
              'progress', 'advance', 'improve', 'enhance', 'strengthen', 'empower', 'enable', 'facilitate', 'help',
              'assist', 'aid', 'benefit', 'advantage', 'gain', 'profit', 'prosper', 'thrive', 'flourish', 'grow'],
              
    'sad': ['defeat', 'lose', 'loss', 'setback', 'disappointment', 'letdown', 'disillusionment', 'disenchantment',
            'disheartened', 'discouraged', 'demoralized', 'deflated', 'downcast', 'downhearted', 'crestfallen',
            'despondent', 'dispirited', 'dejected', 'depressed', 'distressed', 'troubled', 'worried', 'concerned',
            'anxious', 'apprehensive', 'fearful', 'afraid', 'scared', 'terrified', 'horrified', 'alarmed', 'panic'],
            
    'angry': ['oppose', 'resist', 'reject', 'refuse', 'decline', 'deny', 'veto', 'block', 'obstruct', 'impede',
              'hinder', 'hamper', 'thwart', 'foil', 'frustrate', 'disapprove', 'disagree', 'dissent', 'dispute',
              'challenge', 'contest', 'question', 'doubt', 'skeptical', 'cynical', 'critical', 'criticize', 'condemn',
              'denounce', 'decry', 'deplore', 'detest', 'despise', 'loathe', 'abhor', 'hate', 'deride', 'mock', 'scorn'],
              
    'fear': ['threat', 'danger', 'risk', 'hazard', 'peril', 'jeopardy', 'insecurity', 'instability', 'uncertainty',
             'unpredictability', 'volatility', 'turmoil', 'chaos', 'disorder', 'disruption', 'disturbance', 'upheaval',
             'unrest', 'crisis', 'emergency', 'disaster', 'catastrophe', 'calamity', 'tragedy', 'misfortune', 'adversity',
             'hardship', 'difficulty', 'trouble', 'problem', 'issue', 'concern', 'worry', 'anxiety', 'apprehension'],
             
    'surprised': ['unexpected', 'unforeseen', 'unanticipated', 'unpredicted', 'unpredictable', 'surprising', 'shocking',
                 'startling', 'stunning', 'staggering', 'astounding', 'astonishing', 'amazing', 'remarkable', 'extraordinary',
                 'exceptional', 'unprecedented', 'unparalleled', 'unmatched', 'unrivaled', 'unequaled', 'incomparable',
                 'unusual', 'uncommon', 'rare', 'unique', 'special', 'distinctive', 'peculiar', 'odd', 'strange', 'weird'],
                 
    'disgust': ['corrupt', 'corruption', 'scandal', 'controversy', 'misconduct', 'misbehavior', 'wrongdoing', 'impropriety',
               'indiscretion', 'transgression', 'violation', 'infraction', 'breach', 'offense', 'crime', 'felony',
               'misdemeanor', 'illegal', 'unlawful', 'illicit', 'illegitimate', 'improper', 'inappropriate', 'unsuitable',
               'unacceptable', 'objectionable', 'questionable', 'dubious', 'suspicious', 'shady', 'fishy', 'sketchy']
}

def get_emotions(text):
    # Preprocess text
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    
    # Tokenize and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Initialize emotion counts with small baseline values to increase sensitivity
    emotion_count = {emotion: 0.1 for emotion in emotion_keywords}  # Start with a small baseline
    
    # Check for individual words with higher sensitivity
    for token in lemmatized_tokens:
        for emotion, keywords in emotion_keywords.items():
            if token in keywords:
                emotion_count[emotion] += 1.0
        
        # Check political keywords with slightly lower weight
        for emotion, keywords in political_emotion_keywords.items():
            if token in keywords:
                emotion_count[emotion] += 0.7
    
    # Check for phrases with higher weight
    for emotion, phrases in emotion_phrases.items():
        for phrase in phrases:
            if phrase in text:
                emotion_count[emotion] += 2.5  # Give phrases higher weight
    
    # Check for intensifiers near emotion words
    intensifiers = ['very', 'really', 'so', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 
                   'deeply', 'strongly', 'highly', 'quite', 'rather', 'particularly', 'especially', 'notably',
                   'remarkably', 'exceedingly', 'immensely', 'enormously', 'tremendously', 'extraordinarily']
    
    for i, token in enumerate(tokens):
        if i > 0 and tokens[i-1] in intensifiers:
            for emotion, keywords in emotion_keywords.items():
                if token in keywords:
                    emotion_count[emotion] += 1.5  # Add extra weight for intensified emotions
    
    # Check for negations that might flip emotions
    negations = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'can\'t', 'cannot', 'won\'t', 'wouldn\'t',
                'shouldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
                'neither', 'nor', 'barely', 'hardly', 'scarcely', 'rarely', 'seldom']
    
    for i, token in enumerate(tokens):
        if i > 0 and tokens[i-1] in negations:
            # If a happy word is negated, it might indicate sadness
            if token in emotion_keywords['happy']:
                emotion_count['happy'] -= 1.0
                emotion_count['sad'] += 1.0
            # If a sad word is negated, it might indicate happiness
            elif token in emotion_keywords['sad']:
                emotion_count['sad'] -= 1.0
                emotion_count['happy'] += 1.0
    
    # Check for emotion-laden sentence structures
    sentences = sent_tokenize(text)
    for sentence in sentences:
        # Exclamation marks indicate strong emotion
        if '!' in sentence:
            # Try to determine which emotion based on keywords
            has_emotion = False
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in sentence.lower():
                        emotion_count[emotion] += 1.0
                        has_emotion = True
                        break
                if has_emotion:
                    break
            
            # If no specific emotion found, default to surprise
            if not has_emotion:
                emotion_count['surprised'] += 0.5
        
        # Question marks might indicate surprise or fear
        if '?' in sentence:
            if any(word in sentence.lower() for word in emotion_keywords['fear']):
                emotion_count['fear'] += 0.5
            else:
                emotion_count['surprised'] += 0.3
    
    # Check for ALL CAPS words (shouting) - indicates strong emotion
    original_tokens = word_tokenize(text.replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? '))
    for token in original_tokens:
        if token.isupper() and len(token) > 1:  # Ignore single letter uppercase
            # Try to determine which emotion based on the word
            found_emotion = False
            for emotion, keywords in emotion_keywords.items():
                if token.lower() in keywords:
                    emotion_count[emotion] += 1.5
                    found_emotion = True
                    break
            
            # If no specific emotion found, default to anger or surprise
            if not found_emotion:
                emotion_count['angry'] += 0.7
                emotion_count['surprised'] += 0.3
    
    # Ensure no negative counts
    for emotion in emotion_count:
        emotion_count[emotion] = max(0, emotion_count[emotion])
    
    return emotion_count

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        overall = 'Positive'
    elif compound <= -0.05:
        overall = 'Negative'
    else:
        overall = 'Neutral'
    
    # Use sentiment to boost corresponding emotions
    if overall == 'Positive':
        return overall, scores, ['happy']
    elif overall == 'Negative':
        return overall, scores, ['sad', 'angry', 'fear', 'disgust']
    else:
        return overall, scores, []

def analyze_text(user_text):
    overall_sentiment, scores, boosted_emotions = get_sentiment(user_text)
    print(f"\nOverall Sentiment: {overall_sentiment}")
    print(f"Detailed Scores: {scores}")
    
    emotions = get_emotions(user_text)
    
    # Boost emotions based on sentiment
    for emotion in boosted_emotions:
        if emotion in emotions:
            emotions[emotion] += 0.5
    
    print("\nDetected sub-emotion counts: \n")
    for emotion, count in emotions.items():
        print(f"  {emotion}: {count:.2f}")
    
    # Much lower threshold for detecting emotions
    if all(count < 0.5 for count in emotions.values()):
        print("\nNo specific sub-emotion detected.\n")
        dominant_emotion = 'None'
    else:
        dominant_emotion = max(emotions, key=emotions.get)
        print(f"\nDominant sub-emotion: {dominant_emotion}\n")
    
    return overall_sentiment, emotions

def analyze_excel_data(excel_file, text_column):
    """
    Analyze sentiment and emotions from text data in an Excel file.
    
    Parameters:
    - excel_file: Path to the Excel file
    - text_column: Name of the column containing text to analyze
    """
    if not os.path.exists(excel_file):
        print(f"Error: File '{excel_file}' not found.")
        return
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Check if the specified column exists
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found in the Excel file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Add columns for sentiment and emotions
        df['sentiment'] = None
        for emotion in emotion_keywords:
            df[f'emotion_{emotion}'] = 0
        df['dominant_emotion'] = None
        
        # Analyze each text entry
        print(f"\nAnalyzing {len(df)} entries from '{excel_file}'...\n")
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            if pd.isna(text) or text.strip() == '':
                continue
                
            print(f"Entry {idx+1}: {text[:50]}..." if len(text) > 50 else f"Entry {idx+1}: {text}")
            sentiment, emotions = analyze_text(text)
            
            # Store results in dataframe
            df.at[idx, 'sentiment'] = sentiment
            for emotion, count in emotions.items():
                df.at[idx, f'emotion_{emotion}'] = count
            
            # Much lower threshold for detecting emotions
            if all(count < 0.5 for count in emotions.values()):
                df.at[idx, 'dominant_emotion'] = 'None'
            else:
                df.at[idx, 'dominant_emotion'] = max(emotions, key=emotions.get)
        
        # Save results to a new Excel file
        output_file = f"analyzed_{os.path.basename(excel_file)}"
        df.to_excel(output_file, index=False)
        print(f"\nAnalysis complete! Results saved to '{output_file}'")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        sentiment_counts = df['sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        emotion_counts = df['dominant_emotion'].value_counts()
        print("\nDominant Emotion Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
            
    except Exception as e:
        print(f"Error analyzing Excel file: {str(e)}")

if __name__ == "__main__":
    print("Sentiment and Emotion Analysis Tool")
    print("1. Analyze text input")
    print("2. Analyze Excel file")
    
    choice = input("\nSelect an option (1 or 2): ")
    
    if choice == '1':
        while True: 
            user_text = input("Enter text to analyze (type 'exit' to quit): ")
            if user_text.lower() == "exit":
                print("Terminating analysis...")
                break
            else:
                analyze_text(user_text)
    
    elif choice == '2':
        excel_file = input("Enter the path to your Excel file (e.g., 'us_election_2024.xlsx'): ")
        text_column = input("Enter the name of the column containing text to analyze: ")
        analyze_excel_data(excel_file, text_column)
    
    else:
        print("Invalid option selected.")
    
    print("Analysis complete!")
