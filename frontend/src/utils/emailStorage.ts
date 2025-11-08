/**
 * Email verification persistence utilities
 * Stores verified email in localStorage with expiry mechanism
 */

const EMAIL_STORAGE_KEY = 'verified_email';
const EMAIL_TIMESTAMP_KEY = 'verified_email_timestamp';
const VERIFICATION_EXPIRY_HOURS = 24; // Email verification expires after 24 hours

/**
 * Save verified email to localStorage with timestamp
 */
export const saveVerifiedEmail = (email: string): void => {
  localStorage.setItem(EMAIL_STORAGE_KEY, email);
  localStorage.setItem(EMAIL_TIMESTAMP_KEY, Date.now().toString());
};

/**
 * Get verified email from localStorage if still valid
 * Returns null if expired or not found
 */
export const getVerifiedEmail = (): string | null => {
  const email = localStorage.getItem(EMAIL_STORAGE_KEY);
  const timestamp = localStorage.getItem(EMAIL_TIMESTAMP_KEY);

  if (!email || !timestamp) {
    return null;
  }

  // Check if verification has expired
  const hoursSinceVerification = (Date.now() - parseInt(timestamp)) / (1000 * 60 * 60);
  if (hoursSinceVerification > VERIFICATION_EXPIRY_HOURS) {
    clearVerifiedEmail();
    return null;
  }

  return email;
};

/**
 * Clear verified email from localStorage
 */
export const clearVerifiedEmail = (): void => {
  localStorage.removeItem(EMAIL_STORAGE_KEY);
  localStorage.removeItem(EMAIL_TIMESTAMP_KEY);
};

/**
 * Check if email verification is still valid
 */
export const isEmailVerified = (): boolean => {
  return getVerifiedEmail() !== null;
};
