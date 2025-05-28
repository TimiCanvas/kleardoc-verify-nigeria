
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Shield, 
  CheckCircle, 
  AlertCircle, 
  Building2, 
  Search,
  User,
  Calendar,
  MapPin,
  Clock
} from 'lucide-react';
import { useToast } from "@/hooks/use-toast";

const Institutions = () => {
  const [verificationToken, setVerificationToken] = useState('');
  const [verificationResult, setVerificationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // Mock token verification for institutions
  const handleTokenVerification = async () => {
    if (!verificationToken.trim()) {
      toast({
        title: "Token Required",
        description: "Please enter a verification token",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      // Mock verification logic
      const isValidToken = verificationToken.startsWith('klr-') && verificationToken.length > 10;
      
      if (isValidToken) {
        setVerificationResult({
          status: 'verified',
          user: {
            id: 'usr_12345',
            fullName: 'JOHN ADEBAYO DOE',
            ninNumber: '12345***901', // Partially masked for privacy
            verifiedAt: new Date().toISOString(),
            documentType: 'NIN Slip',
            verificationLevel: 'Level 2 - Government ID'
          },
          metadata: {
            verificationDate: new Date().toISOString(),
            confidence: 95.8,
            checks: ['OCR Extraction', 'NIMC Database', 'AI Validation']
          }
        });
        toast({
          title: "Verification Complete",
          description: "User identity has been successfully verified",
        });
      } else {
        setVerificationResult({
          status: 'invalid',
          error: 'Token not found or expired'
        });
        toast({
          title: "Verification Failed",
          description: "Invalid or expired token",
          variant: "destructive",
        });
      }
      
      setIsLoading(false);
    }, 2000);
  };

  const resetVerification = () => {
    setVerificationToken('');
    setVerificationResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <Building2 className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-blue-800">KlearDoc</h1>
                <p className="text-sm text-gray-600">Institutional Portal</p>
              </div>
            </div>
            <Badge variant="outline" className="text-blue-700">
              Financial Institution Access
            </Badge>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Verification Form */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Search className="h-5 w-5" />
                  <span>Token Verification</span>
                </CardTitle>
                <CardDescription>
                  Enter a KlearDoc verification token to validate user identity
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="token">Verification Token</Label>
                    <Input 
                      id="token"
                      value={verificationToken}
                      onChange={(e) => setVerificationToken(e.target.value)}
                      placeholder="klr-xxxxxxxxx"
                      className="font-mono"
                    />
                  </div>
                  
                  <div className="flex space-x-3">
                    <Button 
                      onClick={handleTokenVerification}
                      disabled={isLoading}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {isLoading ? (
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                          <span>Verifying...</span>
                        </div>
                      ) : (
                        'Verify Token'
                      )}
                    </Button>
                    
                    {verificationResult && (
                      <Button 
                        onClick={resetVerification}
                        variant="outline"
                      >
                        New Verification
                      </Button>
                    )}
                  </div>
                </div>

                {/* Verification Results */}
                {verificationResult && (
                  <div className="space-y-4">
                    {verificationResult.status === 'verified' ? (
                      <Alert className="border-green-200 bg-green-50">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                        <AlertDescription>
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <strong className="text-green-800">Identity Verified</strong>
                              <Badge className="bg-green-100 text-green-800">
                                {verificationResult.user.verificationLevel}
                              </Badge>
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                              <div className="space-y-2">
                                <div className="flex items-center space-x-2">
                                  <User className="h-4 w-4 text-gray-600" />
                                  <span className="font-medium">Full Name:</span>
                                </div>
                                <div className="pl-6 text-green-800 font-medium">
                                  {verificationResult.user.fullName}
                                </div>
                              </div>
                              
                              <div className="space-y-2">
                                <div className="flex items-center space-x-2">
                                  <Shield className="h-4 w-4 text-gray-600" />
                                  <span className="font-medium">NIN:</span>
                                </div>
                                <div className="pl-6 text-green-800 font-mono">
                                  {verificationResult.user.ninNumber}
                                </div>
                              </div>
                              
                              <div className="space-y-2">
                                <div className="flex items-center space-x-2">
                                  <Calendar className="h-4 w-4 text-gray-600" />
                                  <span className="font-medium">Verified:</span>
                                </div>
                                <div className="pl-6 text-green-800">
                                  {new Date(verificationResult.user.verifiedAt).toLocaleDateString()}
                                </div>
                              </div>
                              
                              <div className="space-y-2">
                                <div className="flex items-center space-x-2">
                                  <Clock className="h-4 w-4 text-gray-600" />
                                  <span className="font-medium">Confidence:</span>
                                </div>
                                <div className="pl-6 text-green-800 font-medium">
                                  {verificationResult.metadata.confidence}%
                                </div>
                              </div>
                            </div>
                            
                            <div className="pt-2 border-t border-green-200">
                              <div className="text-xs text-green-700">
                                <strong>Verification Checks:</strong> {verificationResult.metadata.checks.join(', ')}
                              </div>
                            </div>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ) : (
                      <Alert className="border-red-200 bg-red-50">
                        <AlertCircle className="h-4 w-4 text-red-600" />
                        <AlertDescription className="text-red-800">
                          <strong>Verification Failed</strong>
                          <div className="mt-2 text-sm">
                            {verificationResult.error}
                          </div>
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Information Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">API Integration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-sm space-y-2">
                  <div className="font-medium">Endpoint:</div>
                  <div className="bg-gray-100 p-2 rounded font-mono text-xs">
                    POST /api/verify-token
                  </div>
                </div>
                
                <div className="text-sm space-y-2">
                  <div className="font-medium">Request Body:</div>
                  <div className="bg-gray-100 p-2 rounded font-mono text-xs">
                    {`{ "token": "klr-xxx..." }`}
                  </div>
                </div>
                
                <div className="text-xs text-gray-600">
                  Contact support for API credentials and integration documentation.
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Verification Levels</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Badge className="bg-green-100 text-green-800">Level 2 - Government ID</Badge>
                  <div className="text-xs text-gray-600">
                    NIN slip verified against NIMC database
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Badge className="bg-blue-100 text-blue-800">Level 1 - Basic</Badge>
                  <div className="text-xs text-gray-600">
                    Utility bills and basic documents
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Security Features</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <FeatureItem 
                  title="Token Expiry" 
                  description="24-hour validity period"
                />
                <FeatureItem 
                  title="Rate Limiting" 
                  description="API call limits per institution"
                />
                <FeatureItem 
                  title="Audit Logs" 
                  description="Complete verification history"
                />
                <FeatureItem 
                  title="Data Privacy" 
                  description="Minimal data exposure"
                />
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

const FeatureItem = ({ title, description }: { title: string; description: string }) => (
  <div className="space-y-1">
    <div className="font-medium text-sm">{title}</div>
    <div className="text-xs text-gray-600">{description}</div>
  </div>
);

export default Institutions;
