

# Use Java 20 instead of Java 17
# This was too brittle: FROM eclipse-temurin:20-jdk
FROM docker.io/library/eclipse-temurin:20-jdk

# Set working directory inside the container
WORKDIR /app

# Copy the compiled JAR from your Gradle build
COPY app/build/libs/client-server.jar app.jar

# Expose port 8080 for WebSocket communication
EXPOSE 8080

# Run the Kotlin server
CMD ["java", "-jar", "app.jar"]
