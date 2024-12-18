I've created a comprehensive dashboard for visualizing LLM probe data with the following features:

Chat Interface:

Real-time chat window with user and assistant messages
Clean, modern design with message bubbles
Smooth scrolling and message history
Moral Dimensions Panel:

Live-updating gauges for each moral dimension
Color-coded variables (Harm: Red, Fairness: Green, Loyalty: Blue, Authority: Yellow, Purity: Purple)
Percentage display for precise readings
Timeline Visualization:

Stacked bar chart showing the evolution of moral dimensions over time
Limited to last 50 data points for performance
Smooth transitions between updates
Layout:

Three-column layout with chat taking 2/3 of the space
Right panel containing moral gauges and timeline
Responsive design that works well on larger screens
The code is structured into reusable components and includes TypeScript types for type safety. The analyzeWithProbes function is currently a placeholder that generates random values - you'll need to replace it with your actual probe implementation.

To integrate your actual probes:

Replace the analyzeWithProbes function with your probe implementation
Update the chat interface to use your LLM instead of the simulated response
Add any additional probe-specific configuration or preprocessing
The dashboard will automatically update as new messages are processed, providing real-time insights into the moral dimensions of the conversation.

Start: npm install
Run: npm run dev
