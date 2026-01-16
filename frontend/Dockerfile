FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# on exporte juste les fichiers build via un stage "artifact"
FROM alpine:3.20
WORKDIR /dist
COPY --from=build /app/dist/ /dist/
