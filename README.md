
# KlearDoc - Nigerian Document Verification Platform

A comprehensive document verification platform designed specifically for Nigeria, featuring AI-powered OCR, NIMC database integration, and secure institutional verification.

## 🚀 Features

### For Users
- **Secure Authentication**: User registration and login system
- **Document Upload**: Support for NIN slips, utility bills, and other Nigerian documents
- **AI-Powered OCR**: Azure AI Vision integration for text extraction
- **NIMC Verification**: Real-time verification against Nigerian identity database
- **AI Assistant**: Intelligent feedback and guidance through verification process
- **Secure Tokens**: Generation of verification tokens for institutional use

### For Financial Institutions
- **Token Verification**: Validate user identities using secure tokens
- **API Integration**: RESTful endpoints for system integration
- **Verification Levels**: Different tiers of identity verification
- **Audit Trails**: Complete verification history and logging

## 🛠 Technology Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **shadcn/ui** components
- **Lucide React** icons
- **React Router** for navigation

### Planned Backend Integration
- **Azure AI Vision** for OCR processing
- **Azure OpenAI** for AI assistant
- **NIMC API** integration
- **SQLite/Firebase** for data storage
- **FastAPI** for backend services

## 🏗 Architecture

```
KlearDoc Platform
├── User Portal (React/TypeScript)
│   ├── Authentication System
│   ├── Document Upload & OCR
│   ├── AI Assistant Integration
│   └── Verification Dashboard
│
├── Institution Portal
│   ├── Token Verification
│   ├── API Access
│   └── Verification Reports
│
└── Backend Services (Planned)
    ├── Azure AI Vision OCR
    ├── NIMC Database Integration
    ├── Azure OpenAI Assistant
    └── Secure Token Management
```

## 🚦 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Modern web browser

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd kleardoc-verify-nigeria

# Install dependencies
npm install

# Start development server
npm run dev
```

### Usage

#### User Portal (Main Application)
1. Visit the main application at `/`
2. Sign up or log in with your credentials
3. Upload a Nigerian document (NIN slip, utility bill)
4. Follow the AI-guided verification process
5. Receive your verification token upon successful verification

#### Institution Portal
1. Visit `/institutions` for the institutional portal
2. Enter a verification token to validate user identity
3. View detailed verification results and metadata

## 🔧 Configuration

### Environment Variables (For Production)
```bash
# Azure Services
AZURE_AI_VISION_ENDPOINT=your_endpoint
AZURE_AI_VISION_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key

# NIMC Integration
NIMC_API_ENDPOINT=your_endpoint
NIMC_API_KEY=your_key

# Database
DATABASE_URL=your_database_url

# Security
JWT_SECRET=your_jwt_secret
TOKEN_ENCRYPTION_KEY=your_encryption_key
```

## 📱 User Experience Flow

### Document Verification Process
1. **Upload**: User uploads Nigerian document
2. **OCR Processing**: Azure AI Vision extracts text
3. **AI Analysis**: OpenAI validates and provides feedback
4. **NIMC Verification**: Cross-check with government database
5. **Token Generation**: Create secure verification token
6. **Institutional Access**: Financial institutions verify tokens

### AI Assistant Features
- Real-time feedback on document quality
- Guidance for document improvements
- Status updates throughout verification
- Natural language explanations of issues

## 🔒 Security Features

- **Secure Token Generation**: UUID-based verification tokens
- **Data Encryption**: All sensitive data encrypted at rest
- **Rate Limiting**: API protection against abuse
- **Audit Logging**: Complete verification trail
- **Privacy Protection**: Minimal data exposure to institutions

## 🌍 Nigerian Market Focus

### Supported Documents
- **National Identity Number (NIN) Slips**
- **Utility Bills** (NEPA, Water, etc.)
- **Bank Statements**
- **Voter Registration Cards**
- **Driver's Licenses**

### Language Support
- English (primary)
- Nigerian Pidgin
- Yoruba
- Hausa
- Igbo

## 📊 Verification Levels

### Level 1 - Basic Verification
- Utility bills and basic documents
- OCR extraction and basic validation
- Suitable for low-risk transactions

### Level 2 - Government ID Verification
- NIN slip verification with NIMC database
- High confidence AI validation
- Suitable for financial services and high-value transactions

## 🔌 API Integration

### Institution API Endpoints
```bash
# Verify a token
POST /api/verify-token
{
  "token": "klr-xxxxxxxxx"
}

# Response
{
  "status": "verified",
  "user": {
    "fullName": "JOHN ADEBAYO DOE",
    "ninNumber": "12345***901",
    "verificationLevel": "Level 2 - Government ID"
  },
  "metadata": {
    "confidence": 95.8,
    "verifiedAt": "2024-01-15T10:30:00Z"
  }
}
```

## 🚀 Deployment

### Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm run preview
```

### Cloud Deployment
- **Vercel/Netlify**: Frontend deployment
- **Azure App Service**: Full-stack deployment
- **Supabase**: Database and authentication
- **Azure Functions**: Serverless backend

## 📈 Future Enhancements

- **Mobile App**: React Native application
- **Blockchain Integration**: Immutable verification records
- **Multi-factor Authentication**: Enhanced security
- **Machine Learning**: Fraud detection algorithms
- **Biometric Verification**: Facial recognition integration
- **API Marketplace**: Third-party integrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For technical support or business inquiries:
- Email: support@kleardoc.com
- Documentation: [docs.kleardoc.com](https://docs.kleardoc.com)
- API Support: api-support@kleardoc.com

## 🏆 Acknowledgments

- **Azure AI Services** for OCR and AI capabilities
- **NIMC** for identity verification infrastructure
- **shadcn/ui** for beautiful UI components
- **Nigerian Fintech Community** for feedback and requirements

---

**KlearDoc** - Securing Nigeria's Digital Identity Future 🇳🇬
