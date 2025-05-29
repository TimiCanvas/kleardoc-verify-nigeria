import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { 
  Shield, 
  Upload, 
  CheckCircle, 
  AlertCircle, 
  FileText, 
  Brain, 
  Key,
  Users,
  Camera,
  Database,
  Zap,
  CreditCard,
  Receipt
} from 'lucide-react';
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [activeTab, setActiveTab] = useState('login');
  const [selectedDocumentType, setSelectedDocumentType] = useState(null);
  const [verificationStep, setVerificationStep] = useState(1);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [ocrData, setOcrData] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState(null);
  const [aiMessage, setAiMessage] = useState('');
  const [verificationToken, setVerificationToken] = useState('');
  const { toast } = useToast();

  // Mock authentication
  const handleLogin = (email: string, password: string) => {
    if (email && password) {
      setCurrentUser({ email, name: 'John Doe' });
      setIsLoggedIn(true);
      toast({
        title: "Login Successful",
        description: "Welcome to KlearDoc!",
      });
    }
  };

  const handleSignup = (name: string, email: string, password: string) => {
    if (name && email && password) {
      setCurrentUser({ email, name });
      setIsLoggedIn(true);
      toast({
        title: "Account Created",
        description: "Welcome to KlearDoc! Your account has been created successfully.",
      });
    }
  };

  // Mock file upload and OCR processing
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>, documentType: string) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setSelectedDocumentType(documentType);
      setVerificationStep(2);
      
      // Simulate OCR processing with different data based on document type
      setTimeout(() => {
        let mockOcrData;
        
        if (documentType === 'nin') {
          mockOcrData = {
            fullName: "JOHN ADEBAYO DOE",
            ninNumber: "12345678901",
            dateOfBirth: "15/03/1990",
            address: "123 VICTORIA ISLAND, LAGOS STATE",
            confidence: 0.95,
            documentType: "NIN Slip",
            rawText: "FEDERAL REPUBLIC OF NIGERIA\nNATIONAL IDENTITY MANAGEMENT COMMISSION\nNATIONAL IDENTITY NUMBER: 12345678901\nSURNAME: DOE\nFIRST NAME: JOHN\nMIDDLE NAME: ADEBAYO\nDATE OF BIRTH: 15/03/1990\nADDRESS: 123 VICTORIA ISLAND, LAGOS STATE"
          };
        } else {
          mockOcrData = {
            fullName: "JOHN ADEBAYO DOE",
            address: "123 VICTORIA ISLAND, LAGOS STATE",
            billType: "ELECTRICITY BILL",
            provider: "EKEDC",
            billDate: "DECEMBER 2024",
            accountNumber: "ACC123456789",
            confidence: 0.92,
            documentType: "Utility Bill",
            rawText: "EKEDC - ELECTRICITY BILL\nCustomer Name: JOHN ADEBAYO DOE\nAddress: 123 VICTORIA ISLAND, LAGOS STATE\nAccount Number: ACC123456789\nBill Period: DECEMBER 2024\nAmount Due: ₦15,000"
          };
        }
        
        setOcrData(mockOcrData);
        setVerificationStep(3);
        setAiMessage(`I've successfully extracted your information from the ${documentType === 'nin' ? 'NIN slip' : 'utility bill'}. The text quality is excellent with ${(mockOcrData.confidence * 100).toFixed(1)}% confidence. Let me now verify this information...`);
      }, 2000);
    }
  };

  // Mock NIMC verification
  const handleNimcVerification = () => {
    setVerificationStep(4);
    
    setTimeout(() => {
      const isVerified = Math.random() > 0.3; // 70% success rate for demo
      
      if (isVerified) {
        setVerificationStatus('approved');
        const token = `klr-${Math.random().toString(36).substr(2, 12)}`;
        setVerificationToken(token);
        setAiMessage("🎉 Excellent! Your identity has been successfully verified. All information matches perfectly. Your verification token has been generated and is ready for use by financial institutions.");
      } else {
        setVerificationStatus('rejected');
        setAiMessage("❌ I found some discrepancies in your document. Please ensure your document is clear and try uploading a newer copy.");
      }
      setVerificationStep(5);
    }, 3000);
  };

  const resetVerification = () => {
    setSelectedDocumentType(null);
    setVerificationStep(1);
    setUploadedFile(null);
    setOcrData(null);
    setVerificationStatus(null);
    setAiMessage('');
    setVerificationToken('');
  };

  // Authentication UI
  const AuthenticationUI = () => (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-white flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex items-center justify-center mb-4">
            <Shield className="h-12 w-12 text-green-600" />
          </div>
          <CardTitle className="text-2xl font-bold text-green-800">KlearDoc</CardTitle>
          <CardDescription>
            Secure Nigerian Document Verification Platform
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">Login</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>
            
            <TabsContent value="login" className="space-y-4">
              <LoginForm onLogin={handleLogin} />
            </TabsContent>
            
            <TabsContent value="signup" className="space-y-4">
              <SignupForm onSignup={handleSignup} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );

  // Main Dashboard
  const Dashboard = () => (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-green-600" />
              <h1 className="text-2xl font-bold text-green-800">KlearDoc</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="text-green-700">
                {currentUser?.name}
              </Badge>
              <Button 
                variant="outline" 
                onClick={() => setIsLoggedIn(false)}
              >
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Document Selection & Verification */}
          <div className="lg:col-span-2">
            {!selectedDocumentType ? (
              <DocumentTypeSelection onSelect={setSelectedDocumentType} />
            ) : (
              <VerificationFlow 
                documentType={selectedDocumentType}
                step={verificationStep}
                onFileUpload={handleFileUpload}
                onVerify={handleNimcVerification}
                onReset={resetVerification}
                ocrData={ocrData}
                verificationStatus={verificationStatus}
                token={verificationToken}
              />
            )}
          </div>

          {/* AI Assistant & Info */}
          <div className="space-y-6">
            {/* AI Assistant */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-blue-600" />
                  <span>AI Assistant</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-blue-800">
                    {aiMessage || "Hello! I'm your AI verification assistant. Select a document type to get started, and I'll guide you through the process."}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Features */}
            <Card>
              <CardHeader>
                <CardTitle>Platform Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <FeatureItem 
                  icon={Camera} 
                  title="Advanced OCR" 
                  description="Azure AI Vision extraction"
                />
                <FeatureItem 
                  icon={Database} 
                  title="NIMC Integration" 
                  description="Real-time database verification"
                />
                <FeatureItem 
                  icon={Brain} 
                  title="AI Guidance" 
                  description="Smart feedback and assistance"
                />
                <FeatureItem 
                  icon={Key} 
                  title="Secure Tokens" 
                  description="Institutional verification"
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );

  return isLoggedIn ? <Dashboard /> : <AuthenticationUI />;
};

// Document Type Selection Component
const DocumentTypeSelection = ({ onSelect }: { onSelect: (type: string) => void }) => (
  <div className="space-y-6">
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Select Document Type</h2>
      <p className="text-gray-600">Choose the type of Nigerian document you want to verify</p>
    </div>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* NIN Documents */}
      <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => onSelect('nin')}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <CreditCard className="h-6 w-6 text-green-600" />
            <span>National Identity Number (NIN)</span>
          </CardTitle>
          <CardDescription>
            Upload your NIN slip for government ID verification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-sm text-gray-600">
              <strong>Supported formats:</strong> JPG, PNG, PDF
            </div>
            <div className="text-sm text-gray-600">
              <strong>Verification level:</strong> Level 2 - Government ID
            </div>
            <Button className="w-full bg-green-600 hover:bg-green-700">
              Select NIN Verification
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Utility Bills */}
      <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => onSelect('utility')}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <Receipt className="h-6 w-6 text-blue-600" />
            <span>Utility Bills</span>
          </CardTitle>
          <CardDescription>
            Upload electricity, water, or gas bills for address verification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-sm text-gray-600">
              <strong>Supported:</strong> NEPA, EKEDC, Water Corporation bills
            </div>
            <div className="text-sm text-gray-600">
              <strong>Verification level:</strong> Level 1 - Address Verification
            </div>
            <Button className="w-full bg-blue-600 hover:bg-blue-700">
              Select Utility Bill
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  </div>
);

// Login Form Component
const LoginForm = ({ onLogin }: { onLogin: (email: string, password: string) => void }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <form onSubmit={(e) => { e.preventDefault(); onLogin(email, password); }} className="space-y-4">
      <div>
        <Label htmlFor="email">Email</Label>
        <Input 
          id="email" 
          type="email" 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email"
          required
        />
      </div>
      <div>
        <Label htmlFor="password">Password</Label>
        <Input 
          id="password" 
          type="password" 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter your password"
          required
        />
      </div>
      <Button type="submit" className="w-full bg-green-600 hover:bg-green-700">
        Login
      </Button>
    </form>
  );
};

// Signup Form Component
const SignupForm = ({ onSignup }: { onSignup: (name: string, email: string, password: string) => void }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <form onSubmit={(e) => { e.preventDefault(); onSignup(name, email, password); }} className="space-y-4">
      <div>
        <Label htmlFor="name">Full Name</Label>
        <Input 
          id="name" 
          type="text" 
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter your full name"
          required
        />
      </div>
      <div>
        <Label htmlFor="signup-email">Email</Label>
        <Input 
          id="signup-email" 
          type="email" 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email"
          required
        />
      </div>
      <div>
        <Label htmlFor="signup-password">Password</Label>
        <Input 
          id="signup-password" 
          type="password" 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Create a password"
          required
        />
      </div>
      <Button type="submit" className="w-full bg-green-600 hover:bg-green-700">
        Create Account
      </Button>
    </form>
  );
};

// Updated Verification Flow Component
const VerificationFlow = ({ 
  documentType,
  step, 
  onFileUpload, 
  onVerify, 
  onReset,
  ocrData, 
  verificationStatus, 
  token 
}: any) => {
  const documentTitle = documentType === 'nin' ? 'NIN Slip' : 'Utility Bill';
  const documentIcon = documentType === 'nin' ? CreditCard : Receipt;
  const IconComponent = documentIcon;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <IconComponent className="h-5 w-5" />
            <span>{documentTitle} Verification</span>
          </CardTitle>
          <Button variant="outline" size="sm" onClick={onReset}>
            Change Document Type
          </Button>
        </div>
        <CardDescription>
          Upload your {documentTitle.toLowerCase()} for verification
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Verification Progress</span>
              <span>{Math.min(step * 20, 100)}%</span>
            </div>
            <Progress value={Math.min(step * 20, 100)} className="h-2" />
          </div>

          {/* Step Content */}
          {step === 1 && (
            <div className="text-center space-y-4">
              <Upload className="h-16 w-16 text-gray-400 mx-auto" />
              <div>
                <h3 className="text-lg font-semibold">Upload Your {documentTitle}</h3>
                <p className="text-gray-600">Supported formats: JPG, PNG, PDF</p>
              </div>
              <div>
                <Label htmlFor="file-upload" className="sr-only">Choose file</Label>
                <Input 
                  id="file-upload"
                  type="file" 
                  accept="image/*,.pdf"
                  onChange={(e) => onFileUpload(e, documentType)}
                  className="file:mr-4 file:py-2 file:px-4 file:border-0 file:bg-green-50 file:text-green-700"
                />
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="text-center space-y-4">
              <div className="animate-spin h-16 w-16 border-4 border-green-200 border-t-green-600 rounded-full mx-auto"></div>
              <div>
                <h3 className="text-lg font-semibold">Processing Document...</h3>
                <p className="text-gray-600">Extracting information using Azure AI Vision</p>
              </div>
            </div>
          )}

          {step >= 3 && ocrData && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Extracted Information</h3>
              <div className="bg-gray-50 p-4 rounded-lg space-y-2">
                <div><strong>Document Type:</strong> {ocrData.documentType}</div>
                <div><strong>Full Name:</strong> {ocrData.fullName}</div>
                {ocrData.ninNumber && <div><strong>NIN Number:</strong> {ocrData.ninNumber}</div>}
                {ocrData.dateOfBirth && <div><strong>Date of Birth:</strong> {ocrData.dateOfBirth}</div>}
                <div><strong>Address:</strong> {ocrData.address}</div>
                {ocrData.billType && <div><strong>Bill Type:</strong> {ocrData.billType}</div>}
                {ocrData.provider && <div><strong>Provider:</strong> {ocrData.provider}</div>}
                {ocrData.accountNumber && <div><strong>Account Number:</strong> {ocrData.accountNumber}</div>}
                <div className="flex items-center space-x-2">
                  <strong>Confidence:</strong> 
                  <Badge variant="outline" className="text-green-700">
                    {(ocrData.confidence * 100).toFixed(1)}%
                  </Badge>
                </div>
              </div>
              
              {step === 3 && (
                <Button onClick={onVerify} className="w-full bg-green-600 hover:bg-green-700">
                  Verify Information
                </Button>
              )}
            </div>
          )}

          {step === 4 && (
            <div className="text-center space-y-4">
              <div className="animate-spin h-16 w-16 border-4 border-blue-200 border-t-blue-600 rounded-full mx-auto"></div>
              <div>
                <h3 className="text-lg font-semibold">Verifying Information...</h3>
                <p className="text-gray-600">Cross-checking your information</p>
              </div>
            </div>
          )}

          {step === 5 && verificationStatus && (
            <div className="space-y-4">
              {verificationStatus === 'approved' ? (
                <Alert className="border-green-200 bg-green-50">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    <strong>Verification Successful!</strong>
                    <div className="mt-2 space-y-2">
                      <div>Your verification token:</div>
                      <div className="font-mono text-sm bg-white p-2 rounded border">
                        {token}
                      </div>
                      <div className="text-xs text-gray-600">
                        Financial institutions can use this token to verify your identity.
                      </div>
                    </div>
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert className="border-red-200 bg-red-50">
                  <AlertCircle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">
                    <strong>Verification Failed</strong>
                    <div className="mt-2">
                      Please check your document and try again.
                    </div>
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Feature Item Component
const FeatureItem = ({ icon: Icon, title, description }: any) => (
  <div className="flex items-start space-x-3">
    <Icon className="h-5 w-5 text-green-600 mt-0.5" />
    <div>
      <div className="font-medium text-sm">{title}</div>
      <div className="text-xs text-gray-600">{description}</div>
    </div>
  </div>
);

export default Index;
